"""
agent.py — Core Agent Logic & NLP Module (Person B)
Equipment Rental Decision Agent — Cymonic Hackathon 2026

This module contains the full decision pipeline:
  parse_inquiry()        → LLM-based text parsing with regex fallback
  match_equipment()      → fuzzy equipment lookup from CSV
  match_contractor()     → contractor lookup from CSV
  calculate_score()      → 5-factor weighted scoring engine
  make_decision()        → threshold-based decision routing
  generate_response()    → quote / info-request / review-ticket builder
  generate_reasoning_text() → LLM-powered chain-of-thought explanation
  run_rental_agent()     → single orchestrator function for the UI

Person C/D: call `run_rental_agent(user_input)` and get back a structured dict.
"""

import os
import json
import re
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

# Groq LLM setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _get_active_groq_model(client: Groq | None) -> str:
    """Find the best available model on the user's Groq account."""
    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        return env_model
    if not client:
        return "qwen/qwen3.8-27b"
    preferred = [
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
    ]
    try:
        available = {m.id for m in client.models.list().data}
        for pref in preferred:
            if pref in available:
                return pref
        for m in available:
            if not any(k in m for k in ("whisper", "guard", "orpheus", "safeguard")):
                return m
    except Exception:
        pass
    return "qwen/qwen3.8-27b"


GROQ_MODEL = _get_active_groq_model(groq_client)

# Data paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EQUIPMENT_CSV = DATA_DIR / "equipment.csv"
CONTRACTORS_CSV = DATA_DIR / "contractors.csv"

# Scoring weights — keep in one place so judges can see them tuned live
SCORING_WEIGHTS = {
    "availability":   30,   # Equipment free + meets min rental
    "completeness":   20,   # All required fields provided
    "customer_trust": 25,   # Tier, payment history, tenure
    "liability_risk": 15,   # Equipment value & site complexity
    "timing":         10,   # Lead time / urgency
}

# Decision thresholds (from work-split PDF)
THRESHOLD_AUTO_QUOTE = 80    # score >= 80 → auto-quote
THRESHOLD_REQUEST_INFO = 50  # 50 <= score < 80 → ask for more info
# score < 50 → manual review

# Tier scoring lookup
TIER_SCORES = {
    "Gold":    25,
    "Silver":  18,
    "Bronze":   5,
    "Unrated":  8,
    "Flagged":  0,
}

# Quote constants
DELIVERY_FEE = 75
DAMAGE_WAIVER_PCT = 0.05
TAX_PCT = 0.08
LOYALTY_DISCOUNT = {"Gold": 0.05, "Silver": 0.02, "Bronze": 0.0, "Unrated": 0.0, "Flagged": 0.0}


# ---------------------------------------------------------------------------
# 1. parse_inquiry  — LLM-based parser with regex fallback
# ---------------------------------------------------------------------------

PARSE_SYSTEM_PROMPT = """You are an intelligent parsing assistant for a heavy equipment rental company chatbot.
Analyze the user's message in the context of the conversation history.

1. Classify the user's INTENT into one of:
   - "greeting": Informal greeting or pleasantry (e.g. "Hello", "Hi", "Hey", "Good morning")
   - "general_question": Asking general questions about fleet, machines, pricing, delivery, locations (e.g. "What equipment do you have?", "Do you rent excavators?", "How much is the crane?")
   - "rental_inquiry": Starting a rental request or asking to book a specific machine
   - "providing_details": Supplying specific details (dates, duration, site, license, company name)
   - "gratitude": Acknowledging or saying thanks (e.g. "Thank you", "Sounds good", "Okay thanks")
   - "reset": Asking to start over, clear chat, or cancel

2. Extract structured fields if mentioned or established in context:
   - equipment_requested: specific equipment name/type or null
   - duration_days: integer number of days or null
   - start_date: string date or null
   - site_info: string site condition/access or null
   - urgency: "high" or "normal"
   - contractor_name_mentioned: company name or null
   - license_mentioned: boolean or string or null

3. Multi-turn rules:
   - If a rental detail was established previously and NOT changed by user, PRESERVE IT.
   - If the user explicitly changes equipment or dates, use the NEW value.
   - If user is just saying "Hello" or asking a general question, do NOT invent rental requirements.

Return ONLY a valid JSON object with keys:
  intent, equipment_requested, duration_days, start_date, site_info,
  urgency, contractor_name_mentioned, license_mentioned
Do not include any markdown fences or text outside the JSON object."""


def parse_inquiry(
    text: str,
    chat_history: list[dict] | None = None,
    session_context: dict | None = None,
) -> dict:
    """Parse free-text inquiry into structured fields with multi-turn memory support.

    Tries Groq LLM first with chat history & context; falls back to regex/keyword
    extraction merged with prior session context.
    """
    # Attempt LLM parsing
    parsed = _parse_with_llm(text, chat_history=chat_history, session_context=session_context)
    if parsed is not None:
        parsed = _normalise_parsed(parsed, text)
    else:
        # Fallback: regex/keyword extraction
        parsed = _parse_with_regex(text, session_context=session_context)

    # Context merge: preserve previously confirmed fields if not mentioned/changed in current turn
    if session_context:
        for key in [
            "equipment_requested",
            "duration_days",
            "start_date",
            "site_info",
            "contractor_name_mentioned",
            "license_mentioned",
        ]:
            if parsed.get(key) is None and session_context.get(key) is not None:
                parsed[key] = session_context[key]

    return parsed


def _parse_with_llm(
    text: str,
    chat_history: list[dict] | None = None,
    session_context: dict | None = None,
) -> dict | None:
    """Call Groq LLM with chat history to extract structured fields. Returns None on failure."""
    if groq_client is None:
        return None
    try:
        messages = [{"role": "system", "content": PARSE_SYSTEM_PROMPT}]

        # Inject known context summary if available
        if session_context:
            known = {k: v for k, v in session_context.items() if v is not None}
            if known:
                messages.append({
                    "role": "system",
                    "content": f"Prior inquiry details already known in this session: {json.dumps(known)}.",
                })

        # Append recent chat history (last 6 messages)
        if chat_history:
            for msg in chat_history[-6:]:
                if isinstance(msg, dict) and msg.get("content"):
                    role = msg.get("role", "user")
                    if role not in ("system", "user", "assistant"):
                        role = "user"
                    messages.append({"role": role, "content": str(msg["content"])})

        # Current user input
        messages.append({"role": "user", "content": text})

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return None


def _parse_with_regex(text: str, session_context: dict | None = None) -> dict:
    """Regex/keyword fallback parser with session context support."""
    lower = text.lower()

    # Equipment — keyword match against known names
    # Equipment — keyword match against known names in fleet
    equipment_keywords = {
        "jcb": "JCB 3CX Backhoe",
        "backhoe": "JCB 3CX Backhoe",
        "3cx": "JCB 3CX Backhoe",
        "dozer": "Caterpillar D6 Dozer",
        "bulldozer": "Caterpillar D6 Dozer",
        "d6": "Caterpillar D6 Dozer",
        "caterpillar": "Caterpillar D6 Dozer",
        "excavator": "Komatsu PC210 Excavator",
        "komatsu": "Komatsu PC210 Excavator",
        "pc210": "Komatsu PC210 Excavator",
        "boom lift": "Genie Z-45 Boom Lift",
        "genie": "Genie Z-45 Boom Lift",
        "z-45": "Genie Z-45 Boom Lift",
        "compressor": "Ingersoll Rand Air Compressor",
        "air compressor": "Ingersoll Rand Air Compressor",
        "ingersoll": "Ingersoll Rand Air Compressor",
        "hauler": "Volvo A40G Hauler",
        "volvo": "Volvo A40G Hauler",
        "a40g": "Volvo A40G Hauler",
        "compactor": "Multiquip Plate Compactor",
        "plate compactor": "Multiquip Plate Compactor",
        "multiquip": "Multiquip Plate Compactor",
        "crane": "Grove GMK3050 Crane",
        "grove": "Grove GMK3050 Crane",
        "gmk3050": "Grove GMK3050 Crane",
        "telehandler": "Skytrak 6034 Telehandler",
        "skytrak": "Skytrak 6034 Telehandler",
        "6034": "Skytrak 6034 Telehandler",
        "generator": "Sullair 185 Portable Generator",
        "sullair": "Sullair 185 Portable Generator",
        "portable generator": "Sullair 185 Portable Generator",
    }
    equipment_requested = None
    # Match longer phrases first
    for kw in sorted(equipment_keywords, key=len, reverse=True):
        if kw in lower:
            equipment_requested = equipment_keywords[kw]
            break

    # Duration
    duration_days = None
    dur_match = re.search(r"(\d+)\s*(days?|day)", lower)
    if dur_match:
        duration_days = int(dur_match.group(1))
    else:
        week_match = re.search(r"(\d+)\s*(weeks?|week)", lower)
        if week_match:
            duration_days = int(week_match.group(1)) * 7

    # Start date
    start_date = None
    date_match = re.search(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:,?\s*\d{4})?)",
        lower,
    )
    if date_match:
        start_date = date_match.group(1).strip()
    elif any(w in lower for w in ["tomorrow", "tommorrow", "tmrw", "tomorow", "tommorow"]):
        start_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "next week" in lower:
        start_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    elif "next month" in lower:
        start_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

    # Urgency
    rush_keywords = ["urgent", "urgently", "rush", "asap", "tomorrow morning",
                     "need it today", "tomorrow", "tommorrow", "immediately"]
    urgency = "high" if any(kw in lower for kw in rush_keywords) else "normal"

    # Site info
    site_info = None
    site_match = re.search(r"(tight|narrow|compacted|flat|indoor|outdoor|downtown|level ground)", lower)
    if site_match:
        site_info = site_match.group(1)

    # License mention
    license_keywords = ["cert", "license", "certified", "certification", "heo",
                        "nccco", "cdl", "operator cert", "class a", "class b"]
    license_mentioned = any(kw in lower for kw in license_keywords)

    # Determine intent in regex fallback
    intent = "rental_inquiry"
    if lower in ("hello", "hi", "hey", "good morning", "good afternoon", "greetings", "howdy"):
        intent = "greeting"
    elif lower in ("thanks", "thank you", "thx", "appreciate it", "great thanks", "sounds good", "perfect"):
        intent = "gratitude"
    elif any(kw in lower for kw in [
        "what do you have", "what all", "what equipment", "catalog", "fleet", "where are you",
        "earthmoving", "lifting", "aerial", "compaction", "what do you mean", "what does",
        "why do you need", "how much is", "do you provide", "do you have"
    ]):
        intent = "general_question"

    # Contractor name — hard to regex, leave None
    contractor_name_mentioned = None

    if session_context:
        if equipment_requested is None:
            equipment_requested = session_context.get("equipment_requested")
        if duration_days is None:
            duration_days = session_context.get("duration_days")
        if start_date is None:
            start_date = session_context.get("start_date")
        if site_info is None:
            site_info = session_context.get("site_info")
        if contractor_name_mentioned is None:
            contractor_name_mentioned = session_context.get("contractor_name_mentioned")
        if not license_mentioned:
            license_mentioned = session_context.get("license_mentioned")

    return {
        "intent": intent,
        "equipment_requested": equipment_requested,
        "duration_days": duration_days,
        "start_date": start_date,
        "site_info": site_info,
        "urgency": urgency,
        "contractor_name_mentioned": contractor_name_mentioned,
        "license_mentioned": license_mentioned if license_mentioned else None,
    }


def _normalise_parsed(parsed: dict, original_text: str) -> dict:
    """Ensure all expected keys exist and types are consistent."""
    defaults = {
        "intent": "rental_inquiry",
        "equipment_requested": None,
        "duration_days": None,
        "start_date": None,
        "site_info": None,
        "urgency": "normal",
        "contractor_name_mentioned": None,
        "license_mentioned": None,
    }
    for key, default in defaults.items():
        if key not in parsed or parsed[key] == "":
            parsed[key] = default

    # Strong keyword overrides for intent
    orig_clean = original_text.strip().lower()
    if orig_clean in ("hello", "hi", "hey", "good morning", "good afternoon", "greetings", "howdy"):
        parsed["intent"] = "greeting"
    elif orig_clean in ("thanks", "thank you", "thx", "appreciate it", "great thanks", "sounds good", "perfect"):
        parsed["intent"] = "gratitude"

    # Coerce duration_days to int if present
    if parsed["duration_days"] is not None:
        try:
            parsed["duration_days"] = int(parsed["duration_days"])
        except (ValueError, TypeError):
            parsed["duration_days"] = None

    # Normalise urgency
    if isinstance(parsed.get("urgency"), str):
        parsed["urgency"] = "high" if parsed["urgency"].lower() in ("high", "rush", "urgent") else "normal"
    else:
        parsed["urgency"] = "normal"

    return parsed


# ---------------------------------------------------------------------------
# 2. match_equipment — fuzzy lookup against equipment.csv
# ---------------------------------------------------------------------------

def match_equipment(parsed: dict, equipment_df: pd.DataFrame) -> pd.Series | None:
    """Match the parsed equipment_requested string to a row in equipment_df.

    Uses substring matching against the 'name' column (case-insensitive).
    Returns the best-matching row as a Series, or None.
    """
    requested = parsed.get("equipment_requested")
    if not requested:
        return None

    req_lower = requested.lower()
    best_match = None
    best_score = 0

    for _, row in equipment_df.iterrows():
        name_lower = row["name"].lower()
        # Exact substring match
        if req_lower in name_lower or name_lower in req_lower:
            score = len(name_lower)
            if score > best_score:
                best_score = score
                best_match = row
        else:
            # Token overlap
            req_tokens = set(req_lower.split())
            name_tokens = set(name_lower.split())
            overlap = len(req_tokens & name_tokens)
            if overlap > best_score:
                best_score = overlap
                best_match = row

    return best_match


# ---------------------------------------------------------------------------
# 3. match_contractor — lookup by name in contractors.csv
# ---------------------------------------------------------------------------

def match_contractor(
    parsed: dict,
    contractors_df: pd.DataFrame,
    raw_text: str = "",
    session_context: dict | None = None,
) -> pd.Series | None:
    """Match contractor to a row in contractors_df.

    Strategy:
      1. Use the LLM-extracted contractor_name_mentioned (substring match).
      2. Fallback: scan the raw inquiry text for any known company name.
      3. Memory Fallback: check contractor from session_context.
    Returns matched row or None.
    """
    # Strategy 1: LLM-extracted name
    name = parsed.get("contractor_name_mentioned")
    if name:
        name_lower = name.lower()
        for _, row in contractors_df.iterrows():
            if (name_lower in row["company_name"].lower()
                    or row["company_name"].lower() in name_lower):
                return row

    # Strategy 2: direct text search for known company names
    if raw_text:
        text_lower = raw_text.lower()
        for _, row in contractors_df.iterrows():
            if row["company_name"].lower() in text_lower:
                return row

    # Strategy 3: session context memory
    if session_context:
        ctx_name = session_context.get("contractor_name_mentioned") or session_context.get("contractor_name")
        if ctx_name:
            ctx_lower = str(ctx_name).lower()
            for _, row in contractors_df.iterrows():
                if (ctx_lower in row["company_name"].lower()
                        or row["company_name"].lower() in ctx_lower):
                    return row

    return None


# ---------------------------------------------------------------------------
# 4. calculate_score — 5-factor weighted scoring engine
# ---------------------------------------------------------------------------

def calculate_score(
    parsed: dict,
    equipment: pd.Series | None,
    contractor: pd.Series | None,
) -> dict:
    """Compute the 5-factor qualification scorecard.

    Returns a dict with per-factor breakdowns (score, max, reason)
    and the total score.
    """
    scorecard = {}

    # --- Factor 1: Availability & Fit (max 30) ---
    avail_max = SCORING_WEIGHTS["availability"]
    if equipment is None:
        avail_score = 0
        avail_reason = "Equipment not identified — cannot verify availability."
    else:
        units = int(equipment.get("units_available", 0))
        min_days = int(equipment.get("min_rental_days", 1))
        duration = parsed.get("duration_days")

        if units <= 0:
            avail_score = 0
            avail_reason = f"No units currently available (0/{int(equipment.get('total_units', 0))})."
        elif duration is not None and duration < min_days:
            avail_score = 15
            avail_reason = (
                f"Units available ({units}), but requested duration "
                f"({duration}d) is below minimum ({min_days}d)."
            )
        elif duration is not None:
            avail_score = 30
            avail_reason = (
                f"Units available ({units}/{int(equipment.get('total_units', 0))}), "
                f"duration ({duration}d) meets minimum ({min_days}d)."
            )
        else:
            avail_score = 20
            avail_reason = f"Units available ({units}), but duration not specified."

    scorecard["availability"] = {
        "score": avail_score, "max": avail_max, "reason": avail_reason,
    }

    # --- Factor 2: Criteria Completeness (max 20) ---
    comp_max = SCORING_WEIGHTS["completeness"]
    comp_score = 0
    missing = []

    if parsed.get("duration_days") is not None:
        comp_score += 5
    else:
        missing.append("duration")

    if parsed.get("start_date") is not None:
        comp_score += 5
    else:
        missing.append("start date")

    if parsed.get("site_info") is not None:
        comp_score += 5
    else:
        missing.append("site access info")

    if parsed.get("license_mentioned"):
        comp_score += 5
    else:
        missing.append("license / certification")

    if missing:
        comp_reason = f"Missing: {', '.join(missing)}."
    else:
        comp_reason = "All required fields provided."

    scorecard["completeness"] = {
        "score": comp_score, "max": comp_max, "reason": comp_reason,
    }

    # --- Factor 3: Customer Trust (max 25) ---
    trust_max = SCORING_WEIGHTS["customer_trust"]
    if contractor is None:
        trust_score = 5
        trust_reason = "Contractor not identified — treated as unknown."
    else:
        tier = str(contractor.get("tier", "Unrated"))
        base = TIER_SCORES.get(tier, 8)

        # Payment history modifier
        avg_days = contractor.get("avg_payment_days", 0)
        try:
            avg_days = float(avg_days)
        except (ValueError, TypeError):
            avg_days = 0
        if avg_days > 60:
            base = max(base - 5, 0)
        elif avg_days > 45:
            base = max(base - 2, 0)

        # Tenure modifier (years_in_business)
        years = contractor.get("years_in_business", 0)
        try:
            years = float(years)
        except (ValueError, TypeError):
            years = 0
        if years < 1:
            base = max(base - 3, 0)

        # Insurance status modifier
        ins = str(contractor.get("insurance_valid", "Yes")).strip().lower()
        ins_note = ""
        if ins in ("no", "false", "0"):
            base = max(base - 10, 0)
            ins_note = ", Insurance: INVALID"

        trust_score = min(base, trust_max)
        trust_reason = (
            f"Tier: {tier}, Avg payment: {avg_days:.0f} days, "
            f"In business: {years:.1f} yrs{ins_note}."
        )

    scorecard["customer_trust"] = {
        "score": trust_score, "max": trust_max, "reason": trust_reason,
    }

    # --- Factor 4: Liability & Value Risk (max 15) ---
    liab_max = SCORING_WEIGHTS["liability_risk"]
    if equipment is None:
        liab_score = 5
        liab_reason = "Equipment not identified — moderate default risk."
    else:
        rate = float(equipment.get("daily_rate", 0))
        site = str(equipment.get("site_access_needed", "Standard access"))

        if rate < 200:
            liab_score = 15
        elif rate < 500:
            liab_score = 10
        elif rate < 1000:
            liab_score = 7
        else:
            liab_score = 3

        # Penalise complex site requirements
        complex_keywords = ["survey", "clearance", "load-bearing", "overhead"]
        if any(kw in site.lower() for kw in complex_keywords):
            liab_score = max(liab_score - 3, 0)

        liab_reason = f"Daily rate: ${rate:.0f}, Site: {site}."

    scorecard["liability_risk"] = {
        "score": liab_score, "max": liab_max, "reason": liab_reason,
    }

    # --- Factor 5: Timing Context (max 10) ---
    timing_max = SCORING_WEIGHTS["timing"]
    urgency = parsed.get("urgency", "normal")
    start_date_raw = parsed.get("start_date")

    if urgency == "high" and start_date_raw is None:
        timing_score = 0
        timing_reason = "Rush urgency with no start date — highest risk."
    elif urgency == "high":
        # Try to compute lead time
        lead_days = _compute_lead_days(start_date_raw)
        if lead_days is not None and lead_days <= 1:
            timing_score = 0
            timing_reason = f"Same-day / next-day rush request ({lead_days}d lead) — high scheduling risk."
        elif lead_days is not None and lead_days <= 3:
            timing_score = 5
            timing_reason = f"Short lead time ({lead_days}d) with rush flag."
        else:
            timing_score = 5
            timing_reason = "Rush flagged but lead time is acceptable."
    else:
        lead_days = _compute_lead_days(start_date_raw) if start_date_raw else None
        if lead_days is not None and lead_days > 3:
            timing_score = 10
            timing_reason = f"Standard lead time ({lead_days}d), no rush."
        elif lead_days is not None:
            timing_score = 7
            timing_reason = f"Short lead time ({lead_days}d), but no rush flag."
        else:
            timing_score = 8
            timing_reason = "No rush flag; start date not specified (assumed standard)."

    scorecard["timing"] = {
        "score": timing_score, "max": timing_max, "reason": timing_reason,
    }

    # --- Total ---
    total = sum(factor["score"] for factor in scorecard.values())
    max_total = sum(factor["max"] for factor in scorecard.values())

    scorecard["total"] = total
    scorecard["max_total"] = max_total

    return scorecard


def _compute_lead_days(start_date_raw) -> int | None:
    """Parse a start date string and return days from now. Returns None on failure."""
    if start_date_raw is None:
        return None
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(str(start_date_raw), fuzzy=True)
        if dt is None:
            return None
        delta = (dt.date() - datetime.now().date()).days
        return max(delta, 0)
    except Exception:
        return None


def make_decision(
    total_score: int,
    scorecard: dict | None = None,
    parsed: dict | None = None,
    contractor: pd.Series | None = None,
    equipment: pd.Series | None = None,
) -> str:
    """Map total score and operational context to a decision string.

    Avoids premature MANUAL_REVIEW for customers who are simply in the discovery
    phase and haven't provided their company name or rental dates yet.
    """
    if total_score >= THRESHOLD_AUTO_QUOTE:
        return "AUTO_QUOTE"

    # Red Flag 1: Equipment completely out of stock
    if equipment is not None and int(equipment.get("units_available", 1)) <= 0:
        return "MANUAL_REVIEW"

    # Red Flag 2: Contractor with invalid insurance or high default risk
    if contractor is not None:
        ins = str(contractor.get("insurance_valid", "Yes")).strip().lower()
        tier = str(contractor.get("tier", ""))
        avg_days = float(contractor.get("avg_payment_days", 0) or 0)
        if ins in ("no", "false", "0"):
            return "MANUAL_REVIEW"
        if tier in ("Flagged", "Bronze") and avg_days > 45:
            return "MANUAL_REVIEW"

    # Red Flag 3: High-value asset rush request
    if parsed and parsed.get("urgency") == "high":
        if equipment is not None and float(equipment.get("daily_rate", 0)) >= 1000:
            return "MANUAL_REVIEW"

    # If score is below 50 simply because information is missing (discovery phase)
    if total_score < THRESHOLD_REQUEST_INFO:
        if parsed and (parsed.get("duration_days") is None or contractor is None):
            return "REQUEST_INFO"
        return "MANUAL_REVIEW"

    return "REQUEST_INFO"


# ---------------------------------------------------------------------------
# 6. generate_response — build decision-specific payloads
# ---------------------------------------------------------------------------

def generate_response(
    decision: str,
    scorecard: dict,
    parsed: dict,
    equipment: pd.Series | None,
    contractor: pd.Series | None,
) -> dict:
    """Build the full response dict with decision-specific payload."""
    response = {
        "decision": decision,
        "scorecard": scorecard,
        "parsed_fields": parsed,
        "equipment_name": equipment["name"] if equipment is not None else None,
        "contractor_name": contractor["company_name"] if contractor is not None else None,
    }

    if decision == "AUTO_QUOTE":
        response["quote"] = _build_quote(parsed, equipment, contractor)
    elif decision == "REQUEST_INFO":
        response["missing_info"] = _build_info_request(parsed, equipment)
    elif decision == "MANUAL_REVIEW":
        response["review_ticket"] = _build_review_ticket(scorecard, parsed, equipment, contractor)

    return response


def _build_quote(
    parsed: dict,
    equipment: pd.Series | None,
    contractor: pd.Series | None,
) -> dict:
    """Compute a real quote breakdown."""
    if equipment is None:
        return {"error": "Cannot generate quote — equipment not identified."}

    daily_rate = float(equipment["daily_rate"])
    duration = parsed.get("duration_days", 1) or 1

    # Use weekly rate logic if duration >= 7
    # Since we don't have weekly_rate in CSV, approximate: weekly = daily * 5.5
    if duration >= 7:
        weeks = math.ceil(duration / 7)
        weekly_rate = daily_rate * 5.5  # approximation
        subtotal = weekly_rate * weeks
    else:
        subtotal = daily_rate * duration

    delivery_fee = DELIVERY_FEE
    damage_waiver = round(subtotal * DAMAGE_WAIVER_PCT, 2)

    # Rush surcharge (if urgency is high)
    rush_surcharge = 0.0
    if parsed.get("urgency") == "high":
        rush_surcharge = round(subtotal * 0.10, 2)

    # Loyalty discount
    discount = 0.0
    tier = "Unrated"
    if contractor is not None:
        tier = str(contractor.get("tier", "Unrated"))
    discount_pct = LOYALTY_DISCOUNT.get(tier, 0.0)
    if discount_pct > 0:
        discount = round(subtotal * discount_pct, 2)

    pre_tax = subtotal + delivery_fee + damage_waiver + rush_surcharge - discount
    tax = round(pre_tax * TAX_PCT, 2)
    total = round(pre_tax + tax, 2)

    return {
        "equipment": equipment["name"],
        "daily_rate": daily_rate,
        "duration_days": duration,
        "subtotal": round(subtotal, 2),
        "delivery_fee": delivery_fee,
        "damage_waiver": damage_waiver,
        "rush_surcharge": rush_surcharge,
        "loyalty_discount": discount,
        "tier": tier,
        "pre_tax_total": round(pre_tax, 2),
        "tax": tax,
        "final_total": total,
    }


def _build_info_request(parsed: dict, equipment: pd.Series | None) -> dict:
    """Generate a specific info-request listing exactly which fields are missing."""
    missing_fields = []
    if parsed.get("duration_days") is None:
        missing_fields.append("exact rental duration (number of days)")
    if parsed.get("start_date") is None:
        missing_fields.append("confirmed start date")
    if not parsed.get("license_mentioned"):
        if equipment is not None and pd.notna(equipment.get("required_license")) and equipment.get("required_license"):
            missing_fields.append(
                f"confirmation of your {equipment['required_license']}"
            )
        else:
            missing_fields.append("relevant license or certification details")
    if parsed.get("site_info") is None:
        missing_fields.append("site access details (surface type, clearance, etc.)")

    if not missing_fields:
        missing_fields.append("additional details to verify your request")

    numbered = "\n".join(f"  ({i+1}) {f}" for i, f in enumerate(missing_fields))
    message = (
        f"Thanks for reaching out! To lock in availability and pricing, "
        f"we need a few more details:\n{numbered}\n\n"
        f"Once we have these, we can issue your quote right away."
    )

    return {
        "missing_fields": missing_fields,
        "message": message,
    }


def _build_review_ticket(
    scorecard: dict,
    parsed: dict,
    equipment: pd.Series | None,
    contractor: pd.Series | None,
) -> dict:
    """Generate a manual review ticket with priority and trigger reasons."""
    # Determine priority
    priority = "Urgent" if parsed.get("urgency") == "high" else "Standard"

    # Find low-scoring factors (below 50% of their max)
    triggers = []
    for factor_name in ["availability", "completeness", "customer_trust",
                        "liability_risk", "timing"]:
        factor = scorecard.get(factor_name, {})
        if factor.get("score", 0) < factor.get("max", 1) * 0.5:
            triggers.append(f"{factor_name}: {factor['score']}/{factor['max']} — {factor.get('reason', '')}")

    if not triggers:
        triggers.append("Overall score below threshold for auto-quote.")

    equip_name = equipment["name"] if equipment is not None else "Unknown"
    contr_name = contractor["company_name"] if contractor is not None else "Unknown"

    return {
        "priority": priority,
        "equipment": equip_name,
        "contractor": contr_name,
        "triggers": triggers,
        "recommendation": _recommend_next_step(scorecard, parsed),
    }


def _recommend_next_step(scorecard: dict, parsed: dict) -> str:
    """One-line recommended next step for the reviewer."""
    if scorecard.get("availability", {}).get("score", 0) == 0:
        return "Check availability or suggest alternative equipment."
    if scorecard.get("customer_trust", {}).get("score", 0) < 10:
        return "Verify contractor credentials and review account status."
    if parsed.get("site_info") is None:
        return "Request a site-access survey before quoting."
    return "Review inquiry details and contact contractor for clarification."


# ---------------------------------------------------------------------------
# 7. Conversational Dialogue Generator (Agentic Persona)
# ---------------------------------------------------------------------------

AGENTIC_REPLY_PROMPT = """You are Alex, an experienced and friendly rental operations specialist at Cymonic Heavy Equipment Rentals.
You chat like a real, helpful human customer service agent — warm, professional, clear, and proactive.

Company Fleet:
- JCB 3CX Backhoe: $450/day, 3 available (requires Class A license)
- Caterpillar D6 Dozer: $850/day, 1 available (requires Class A license)
- Komatsu PC210 Excavator: $620/day, 4 available (requires Class B license)
- Genie Z-45 Boom Lift: $320/day, 0 available (currently rented out)
- Ingersoll Rand Air Compressor: $210/day, 5 available
- Volvo A40G Hauler: $1100/day, 1 available (requires Class A license)
- Multiquip Plate Compactor: $150/day, 8 available
- Grove GMK3050 Crane: $1200/day, 2 available (requires Class B license)
- Skytrak 6034 Telehandler: $550/day, 3 available (requires Class B license)
- Sullair 185 Portable Generator: $280/day, 7 available

Your mission:
Write the exact message the assistant should send to the user based on the context and operational state below.
Rules:
1. GREETING: Warmly welcome the customer, introduce yourself as Alex from Cymonic Rentals, and ask how you can help with their job site equipment needs.
2. GENERAL_QUESTION: Answer accurately using our fleet data above. If the customer asked about a machinery category (earthmoving, aerial, lifting, compaction, power), list the specific matching machines with their day rates and availability. If they asked about license requirements (Class A or B), explain clearly that Class A/B refers to heavy equipment operator certifications needed for safe operation. Keep it concise, helpful, and invite them to explore specific machines.
3. REQUEST_INFO: Enthusiastically confirm equipment interest and stock availability. CRITICAL: If the customer asked a question (e.g. what Class A license means, or explained that their start date is tentative / not confirmed), FIRST directly answer their question in a helpful, conversational, and reassuring way. Then conversationally ask for whatever remaining missing details (e.g. rental duration, start date, site access, or contractor company name) are needed so you can issue a firm quote.
4. AUTO_QUOTE: Warmly congratulate the contractor! Clearly summarize the rental terms (machine, duration, start date, contractor name, loyalty discount if any), state the total price clearly, and ask if they'd like to confirm the booking.
5. MANUAL_REVIEW: Be courteous and reassuring. Explain professionally that because of specific operational/safety requirements (such as crane site clearance, heavy hauling route survey, or account verification), our operations team is preparing an urgent review to confirm availability.
6. GRATITUDE: Acknowledge politely and offer further assistance.
7. Keep the response concise, natural, and friendly (2 to 4 sentences). Never include markdown JSON, robot labels, or system headers in your conversational reply."""


def _generate_conversational_reply(
    user_input: str,
    decision: str,
    parsed: dict,
    equipment: pd.Series | None,
    contractor: pd.Series | None,
    scorecard: dict | None,
    chat_history: list[dict] | None = None,
    quote: dict | None = None,
    review_ticket: dict | None = None,
    missing_info: dict | None = None,
) -> str:
    """Generate a warm, natural, human-like dialogue response using Groq LLM (with template fallback)."""
    if groq_client is not None:
        try:
            equip_str = (
                f"{equipment['name']} (${equipment['daily_rate']}/day, {equipment['units_available']} available)"
                if equipment is not None else "None identified"
            )
            contr_str = (
                f"{contractor['company_name']} ({contractor['tier']} tier, insurance={contractor.get('insurance_valid', 'Yes')})"
                if contractor is not None else "Unknown / Not identified"
            )

            state_summary = (
                f"Customer Input: \"{user_input}\"\n"
                f"Intent: {parsed.get('intent', 'rental_inquiry')}\n"
                f"Decision: {decision}\n"
                f"Equipment: {equip_str}\n"
                f"Contractor: {contr_str}\n"
            )

            if quote:
                state_summary += (
                    f"Quote Total: ${quote['final_total']:.2f} "
                    f"(Subtotal: ${quote['subtotal']:.2f}, Daily: ${quote['daily_rate']}, "
                    f"Days: {quote['duration_days']}, Loyalty Discount: ${quote['loyalty_discount']:.2f})\n"
                )
            if missing_info:
                state_summary += f"Missing Fields to ask for: {', '.join(missing_info.get('missing_fields', []))}\n"
            if review_ticket:
                state_summary += f"Review Priority: {review_ticket.get('priority')}, Triggers: {'; '.join(review_ticket.get('triggers', []))}\n"

            messages = [
                {"role": "system", "content": AGENTIC_REPLY_PROMPT},
            ]
            if chat_history:
                for msg in chat_history[-4:]:
                    role = msg.get("role", "user")
                    if role in ("user", "assistant"):
                        messages.append({"role": role, "content": str(msg.get("content", ""))})

            messages.append({
                "role": "user",
                "content": f"Please generate the natural chat reply for this situation:\n{state_summary}",
            })

            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=300,
            )
            reply = response.choices[0].message.content.strip()
            if reply.startswith('"') and reply.endswith('"'):
                reply = reply[1:-1]
            return reply
        except Exception:
            pass

    # High quality human fallback
    if decision in ("GREETING", "greeting"):
        return (
            "Hello! Welcome to Cymonic Heavy Equipment Rentals. My name is Alex. "
            "How can I assist you with your equipment or job site needs today?"
        )
    elif decision in ("GRATITUDE", "gratitude"):
        return "You're very welcome! If you need anything else or want to adjust details, just let me know."
    elif decision in ("GENERAL_QUESTION", "general_question"):
        return (
            "We maintain a full commercial fleet including backhoes, dozers, excavators, "
            "boom lifts, cranes, haulers, and generators. What machine can we quote for you?"
        )
    elif decision == "AUTO_QUOTE" and quote:
        return (
            f"Great news! Your quote for the {quote['equipment']} is confirmed at "
            f"${quote['final_total']:,.2f} for {quote['duration_days']} days. "
            f"We have reserved the unit for you. Would you like to proceed with booking?"
        )
    elif decision == "REQUEST_INFO" and missing_info:
        missing_str = ", ".join(missing_info.get("missing_fields", ["additional details"]))
        equip_name = equipment["name"] if equipment is not None else "equipment"
        lower_in = user_input.lower()
        prefix = ""
        if "class a" in lower_in or "class b" in lower_in or "license" in lower_in:
            prefix = "A Class A license is a heavy equipment operator certification required for safe operation of this machine. "
        elif "not confirmed" in lower_in or "tentative" in lower_in:
            prefix = "No problem if your start date is tentative—we can prepare an initial reservation hold! "

        return (
            f"{prefix}To lock in your reservation and pricing for the {equip_name}, "
            f"could you provide your {missing_str}?"
        )
    elif decision == "MANUAL_REVIEW" and review_ticket:
        return (
            "Thanks for reaching out. Due to specific operational and site clearance requirements, "
            "our operations team is conducting an urgent review of this request. "
            "A rental coordinator will follow up with you shortly."
        )
    elif decision in ("UNRECOGNIZED", "GENERAL_QUESTION", "general_question"):
        lower_in = user_input.lower()
        if "earthmoving" in lower_in or "earth moving" in lower_in:
            return (
                "For earthmoving, we carry the JCB 3CX Backhoe ($450/day), Caterpillar D6 Dozer ($850/day), "
                "Komatsu PC210 Excavator ($620/day), and Volvo A40G Hauler ($1,100/day). Which machine would you like a quote for?"
            )
        elif "aerial" in lower_in or "lift" in lower_in:
            return (
                "For aerial and lifting, we carry the Genie Z-45 Boom Lift ($320/day), "
                "Grove GMK3050 Crane ($1,200/day), and Skytrak 6034 Telehandler ($550/day). Which one fits your project?"
            )
        elif "compaction" in lower_in:
            return "For compaction, we offer the Multiquip Plate Compactor ($150/day, 8 available). How many days will you need it?"
        elif "power" in lower_in or "generator" in lower_in or "compressor" in lower_in:
            return (
                "For power and compressed air, we have the Sullair 185 Portable Generator ($280/day) "
                "and Ingersoll Rand Air Compressor ($210/day). Would you like to reserve one?"
            )
        return (
            "We carry earthmoving, lifting, aerial, compaction, and power generation equipment. "
            "Could you specify the machine name you're looking for, such as our backhoes, dozers, or excavators?"
        )
    else:
        return "I'd be glad to assist! Could you specify what machine you need and when your project begins?"


# ---------------------------------------------------------------------------
# 8. generate_reasoning_text — LLM chain-of-thought explanation
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = """You are drafting an internal decision note for a rental operations team.
Given the scorecard and decision below, write a concise chain-of-thought
reasoning that a human manager would find clear and defensible.
Format it as a numbered list showing each scoring step.
Reference only the factors present in the data. Do not invent facts."""


def generate_reasoning_text(
    scorecard: dict,
    decision: str,
    parsed: dict,
    session_context: dict | None = None,
) -> str:
    """Generate human-readable chain-of-thought reasoning.

    Tries LLM first, falls back to template-based output.
    """
    # Try LLM
    reasoning = _reasoning_with_llm(scorecard, decision, parsed, session_context=session_context)
    if reasoning:
        return reasoning

    # Template fallback
    return _reasoning_template(scorecard, decision, parsed, session_context=session_context)


def _reasoning_with_llm(
    scorecard: dict,
    decision: str,
    parsed: dict,
    session_context: dict | None = None,
) -> str | None:
    """Use Groq LLM to generate reasoning text."""
    if groq_client is None:
        return None

    # Build a compact scorecard summary for the prompt
    factors_summary = []
    for name in ["availability", "completeness", "customer_trust",
                  "liability_risk", "timing"]:
        f = scorecard.get(name, {})
        factors_summary.append(f"{name}: {f.get('score', '?')}/{f.get('max', '?')} — {f.get('reason', '')}")

    context_note = ""
    if session_context:
        retained = [k for k, v in session_context.items() if v is not None and v == parsed.get(k)]
        if retained:
            context_note = f"Context retained from prior message: {', '.join(retained)}\n"

    prompt_data = (
        f"{context_note}"
        f"Scorecard:\n" + "\n".join(factors_summary) + "\n"
        f"Total: {scorecard.get('total', '?')}/{scorecard.get('max_total', '?')}\n"
        f"Decision: {decision}\n"
        f"Equipment requested: {parsed.get('equipment_requested', 'N/A')}\n"
        f"Duration: {parsed.get('duration_days', 'N/A')} days\n"
        f"Urgency: {parsed.get('urgency', 'N/A')}"
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_data},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def _reasoning_template(
    scorecard: dict,
    decision: str,
    parsed: dict,
    session_context: dict | None = None,
) -> str:
    """Template-based reasoning fallback (no LLM needed)."""
    lines = ["🧠 Agent Thought Process:"]
    step = 1

    equip = parsed.get("equipment_requested", "Unknown")
    dur = parsed.get("duration_days", "?")
    start = parsed.get("start_date", "not specified")

    # Check if context was carried forward from prior message
    if session_context:
        retained_items = []
        for k, label in [
            ("equipment_requested", "equipment"),
            ("duration_days", "duration"),
            ("start_date", "start date"),
            ("contractor_name_mentioned", "contractor"),
        ]:
            if session_context.get(k) is not None and session_context.get(k) == parsed.get(k):
                retained_items.append(label)
        if retained_items:
            lines.append(f"  {step}. Context Retained: Preserved {', '.join(retained_items)} from earlier in conversation.")
            step += 1

    lines.append(f"  {step}. Current Inquiry: {equip}, {dur} days, start: {start}.")
    step += 1

    for name, label in [
        ("availability", "Stock Check"),
        ("completeness", "Info Completeness"),
        ("customer_trust", "Contractor Trust"),
        ("liability_risk", "Risk Assessment"),
        ("timing", "Timing Check"),
    ]:
        f = scorecard.get(name, {})
        lines.append(
            f"  {step}. {label}: {f.get('score', '?')}/{f.get('max', '?')} pts"
            f" — {f.get('reason', '')}"
        )
        step += 1

    total = scorecard.get("total", "?")
    max_t = scorecard.get("max_total", "?")

    decision_emoji = {"AUTO_QUOTE": "✅", "REQUEST_INFO": "⚠️", "MANUAL_REVIEW": "🚨"}
    emoji = decision_emoji.get(decision, "❓")
    decision_label = decision.replace("_", " ").title()

    lines.append(f"  Total Score: {total}/{max_t} → Decision: {emoji} {decision_label}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. run_rental_agent — single orchestrator with multi-turn memory
# ---------------------------------------------------------------------------

def run_rental_agent(
    user_input: str,
    chat_history: list[dict] | None = None,
    session_context: dict | None = None,
) -> dict:
    """Main orchestrator: parse → match → score → decide → respond.

    Supports multi-turn memory via `chat_history` and `session_context`.
    This is the ONLY function Person C/D needs to call from the Streamlit UI.

    Args:
        user_input: Raw free-text inquiry from the contractor.
        chat_history: Optional list of past messages, e.g. [{"role": "user", "content": "..."}, ...]
        session_context: Optional dict of previously extracted fields or session state.

    Returns:
        A dict with keys:
          - decision: "AUTO_QUOTE" | "REQUEST_INFO" | "MANUAL_REVIEW" | "UNRECOGNIZED"
          - scorecard: per-factor breakdown dict (if recognized)
          - reasoning: formatted chain-of-thought string
          - parsed_fields: the extracted structured fields
          - equipment_name: matched equipment name or None
          - contractor_name: matched contractor name or None
          - session_context: updated accumulated state across turns
          - is_followup: bool indicating if this turn built upon previous context
          - quote: (only if AUTO_QUOTE) quote breakdown dict
          - missing_info: (only if REQUEST_INFO) info request dict
          - review_ticket: (only if MANUAL_REVIEW) review ticket dict
    """
    if not user_input or not user_input.strip():
        return {
            "decision": "UNRECOGNIZED",
            "message": "It looks like the message was empty. Could you describe what equipment you need?",
            "reasoning": "🧠 Agent Thought Process:\n  1. Received empty input — no analysis possible.",
            "scorecard": None,
            "parsed_fields": {},
            "equipment_name": None,
            "contractor_name": None,
            "session_context": session_context or {},
            "is_followup": False,
        }

    # Check for reset commands
    clean_input = user_input.strip().lower()
    if clean_input in ("reset", "start over", "clear", "clear chat", "new inquiry"):
        return {
            "decision": "UNRECOGNIZED",
            "message": "Session reset! What equipment would you like to inquire about?",
            "reasoning": "🧠 Agent Thought Process:\n  1. User requested session reset. Cleared previous inquiry memory.",
            "scorecard": None,
            "parsed_fields": {},
            "equipment_name": None,
            "contractor_name": None,
            "session_context": {},
            "is_followup": False,
        }

    # 1. Load data
    try:
        equipment_df = pd.read_csv(EQUIPMENT_CSV)
        contractors_df = pd.read_csv(CONTRACTORS_CSV)
    except FileNotFoundError as e:
        return {
            "decision": "UNRECOGNIZED",
            "message": f"Data files not found: {e}. Please ensure CSVs are in the data/ folder.",
            "reasoning": "🧠 System error — data files missing.",
            "scorecard": None,
            "parsed_fields": {},
            "equipment_name": None,
            "contractor_name": None,
            "session_context": session_context or {},
            "is_followup": False,
        }

    # 2. Parse the inquiry with multi-turn support
    parsed = parse_inquiry(user_input, chat_history=chat_history, session_context=session_context)
    intent = parsed.get("intent", "rental_inquiry")

    # Handle pure conversational intents (Greeting, Gratitude, General Questions)
    if intent == "greeting":
        msg = _generate_conversational_reply(user_input, "GREETING", parsed, None, None, None, chat_history=chat_history)
        return {
            "decision": "IN_CONVERSATION",
            "message": msg,
            "reasoning": "🧠 Agent Thought Process:\n  1. Recognized greeting from customer.\n  2. Responded with friendly introductory welcome and offered machinery assistance.",
            "scorecard": None,
            "parsed_fields": parsed,
            "equipment_name": session_context.get("equipment_requested") if session_context else None,
            "contractor_name": session_context.get("contractor_name_mentioned") if session_context else None,
            "session_context": session_context or {},
            "is_followup": bool(session_context),
        }

    if intent == "gratitude":
        msg = _generate_conversational_reply(user_input, "GRATITUDE", parsed, None, None, None, chat_history=chat_history)
        return {
            "decision": "IN_CONVERSATION",
            "message": msg,
            "reasoning": "🧠 Agent Thought Process:\n  1. Customer expressed thanks / acknowledgement.\n  2. Acknowledged politely and affirmed readiness for next steps.",
            "scorecard": None,
            "parsed_fields": parsed,
            "equipment_name": session_context.get("equipment_requested") if session_context else None,
            "contractor_name": session_context.get("contractor_name_mentioned") if session_context else None,
            "session_context": session_context or {},
            "is_followup": bool(session_context),
        }

    if intent == "general_question" and not parsed.get("equipment_requested"):
        msg = _generate_conversational_reply(user_input, "GENERAL_QUESTION", parsed, None, None, None, chat_history=chat_history)
        return {
            "decision": "IN_CONVERSATION",
            "message": msg,
            "reasoning": "🧠 Agent Thought Process:\n  1. Addressed general customer inquiry regarding fleet availability and equipment types.\n  2. Provided concise catalog guidance.",
            "scorecard": None,
            "parsed_fields": parsed,
            "equipment_name": session_context.get("equipment_requested") if session_context else None,
            "contractor_name": session_context.get("contractor_name_mentioned") if session_context else None,
            "session_context": session_context or {},
            "is_followup": bool(session_context),
        }

    # 3. Match equipment and contractor
    equipment = match_equipment(parsed, equipment_df)
    contractor = match_contractor(parsed, contractors_df, raw_text=user_input, session_context=session_context)

    is_followup = bool(
        session_context and any(
            session_context.get(k) is not None
            for k in ["equipment_requested", "duration_days", "start_date"]
        )
    )

    # 4. Handle unrecognized equipment
    if equipment is None:
        msg = _generate_conversational_reply(
            user_input, "UNRECOGNIZED", parsed, None, contractor, None,
            chat_history=chat_history,
        )
        return {
            "decision": "UNRECOGNIZED",
            "message": msg,
            "reasoning": (
                "🧠 Agent Thought Process:\n"
                f"  1. Parsed inquiry but could not identify specific machine in fleet.\n"
                f"  2. Equipment requested: '{parsed.get('equipment_requested', 'None detected')}'\n"
                f"  3. Action: Naturally guided customer toward our available machinery categories."
            ),
            "scorecard": None,
            "parsed_fields": parsed,
            "equipment_name": None,
            "contractor_name": contractor["company_name"] if contractor is not None else None,
            "session_context": session_context or parsed,
            "is_followup": is_followup,
        }

    # 5. Score
    scorecard = calculate_score(parsed, equipment, contractor)

    # 6. Decide
    decision = make_decision(
        scorecard["total"],
        scorecard=scorecard,
        parsed=parsed,
        contractor=contractor,
        equipment=equipment,
    )

    # 7. Generate decision-specific response
    response = generate_response(decision, scorecard, parsed, equipment, contractor)

    # 8. Generate dynamic human message
    response["message"] = _generate_conversational_reply(
        user_input,
        decision,
        parsed,
        equipment,
        contractor,
        scorecard,
        chat_history=chat_history,
        quote=response.get("quote"),
        review_ticket=response.get("review_ticket"),
        missing_info=response.get("missing_info"),
    )

    # 9. Add reasoning & multi-turn metadata
    response["reasoning"] = generate_reasoning_text(scorecard, decision, parsed, session_context=session_context)
    response["session_context"] = parsed
    response["is_followup"] = is_followup

    return response


# ---------------------------------------------------------------------------
# Quick test (run this file directly)
# ---------------------------------------------------------------------------

def run_automated_tests():
    """Run all automated single-turn and multi-turn test cases."""
    test_inputs = [
        # Should -> AUTO_QUOTE (Gold tier, all info, low-risk equipment)
        "Need the JCB 3CX Backhoe for 5 days starting Sept 15, site is confirmed, "
        "our operator holds Class A license. Can you confirm? "
        "This is Smith & Sons Construction.",

        # Should -> REQUEST_INFO (missing dates, license, site)
        "Need a boom lift for a job next week, maybe 4-5 days, not sure on "
        "exact dates yet, will confirm site details soon.",

        # Should -> MANUAL_REVIEW (rush, high-value, bronze contractor with no insurance)
        "Need the Grove GMK3050 Crane for a demolition job starting tomorrow morning, "
        "3 days, downtown site with tight access. Big contract riding on this. "
        "This is Weekend Warriors.",

        # Should -> UNRECOGNIZED
        "Hello, just checking in about the weather today.",

        "Hello Is the chainsaw available"
    ]

    for i, inp in enumerate(test_inputs, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {inp[:80]}...")
        print(f"{'='*70}")
        result = run_rental_agent(inp)
        print(f"Decision: {result['decision']}")
        if result.get("scorecard"):
            sc = result["scorecard"]
            print(f"Score: {sc['total']}/{sc['max_total']}")
            for fname in ["availability", "completeness", "customer_trust",
                          "liability_risk", "timing"]:
                f = sc.get(fname, {})
                print(f"  {fname}: {f.get('score')}/{f.get('max')} - {f.get('reason')}")
        print(f"\nParsed: {result.get('parsed_fields', {})}")
        print(f"Equipment: {result.get('equipment_name')}")
        print(f"Contractor: {result.get('contractor_name')}")
        print(f"\nReasoning:\n{result.get('reasoning', 'N/A')}")
        if result.get("quote"):
            q = result["quote"]
            print(f"\nQuote: ${q['final_total']:.2f}")
            for k, v in q.items():
                print(f"  {k}: {v}")
        if result.get("missing_info"):
            print(f"\nInfo Request:\n{result['missing_info']['message']}")
        if result.get("review_ticket"):
            t = result["review_ticket"]
            print(f"\nReview Ticket: Priority={t['priority']}")
            for tr in t.get("triggers", []):
                print(f"  - {tr}")
        if result.get("message"):
            print(f"\nMessage: {result['message']}")

    # Multi-turn demo test
    print(f"\n{'='*70}")
    print("TEST MULTI-TURN CONVERSATION MEMORY:")
    print(f"{'='*70}")

    turn1_input = "Hi, I need to rent the Caterpillar D6 Dozer for next week."
    print(f"\n[USER Turn 1]: {turn1_input}")
    res1 = run_rental_agent(turn1_input)
    print(f"[AGENT Turn 1]: Decision={res1['decision']}, Equipment={res1['equipment_name']}")
    if res1.get("missing_info"):
        print(f"  Missing: {res1['missing_info']['missing_fields']}")

    history = [
        {"role": "user", "content": turn1_input},
        {"role": "assistant", "content": res1.get("missing_info", {}).get("message", "Need more details.")},
    ]

    turn2_input = "It will be for 4 days starting Sept 22. Site is confirmed, operator holds Class A license. This is Smith & Sons Construction."
    print(f"\n[USER Turn 2]: {turn2_input}")
    res2 = run_rental_agent(
        turn2_input,
        chat_history=history,
        session_context=res1["session_context"],
    )
    print(f"[AGENT Turn 2]: Decision={res2['decision']}, Equipment={res2['equipment_name']}, Contractor={res2['contractor_name']}")
    print(f"Score: {res2.get('scorecard', {}).get('total')}/100")
    if res2.get("quote"):
        print(f"Quote Generated: ${res2['quote']['final_total']:.2f}")
    print(f"Reasoning:\n{res2['reasoning']}")


def interactive_chat():
    """Live interactive chat loop in terminal with multi-turn memory."""
    print("=" * 70)
    print("🏗️  EQUIPMENT RENTAL DECISION AGENT — INTERACTIVE CHAT MODE")
    print("    Type your message to talk with the agent.")
    print("    Commands: 'reset' (clear memory), 'test' (run test suite), 'exit' (quit)")
    print("=" * 70)

    chat_history = []
    session_context = {}

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting chat. Goodbye!")
            break
        if user_input.lower() == "test":
            run_automated_tests()
            continue
        if user_input.lower() in ("reset", "clear"):
            chat_history = []
            session_context = {}
            print("🔄 Session reset! Context memory cleared.")
            continue

        result = run_rental_agent(
            user_input,
            chat_history=chat_history,
            session_context=session_context,
        )

        # Update session memory
        session_context = result.get("session_context", {})

        decision = result.get("decision", "UNKNOWN")
        emoji_map = {
            "AUTO_QUOTE": "✅",
            "REQUEST_INFO": "⚠️",
            "MANUAL_REVIEW": "🚨",
            "UNRECOGNIZED": "❓",
            "IN_CONVERSATION": "💬",
            "GREETING": "👋",
        }

        # 1. Primary human agent dialogue
        print("\n" + "-" * 70)
        print(f"🤖 Alex (Rental Specialist):")
        print(f"   {result.get('message', '')}")

        # 2. Operational assessment card (shown during rental evaluation)
        if decision not in ("IN_CONVERSATION", "GREETING") or result.get("scorecard"):
            print("\n📋 [Operational Assessment]")
            print(f"   Status: {emoji_map.get(decision, '🔹')} {decision}")

            if result.get("scorecard"):
                sc = result["scorecard"]
                print(f"   Qualification Score: {sc.get('total')}/{sc.get('max_total')}")

            if decision == "AUTO_QUOTE" and result.get("quote"):
                q = result["quote"]
                print(f"   💰 Approved Quote: ${q['final_total']:,.2f}")
                print(f"      Machine:  {q['equipment']} (${q['daily_rate']}/day)")
                print(f"      Duration: {q['duration_days']} days (Subtotal: ${q['subtotal']:,.2f})")
                if q.get("loyalty_discount", 0) > 0:
                    print(f"      Discount: -${q['loyalty_discount']:.2f} ({q.get('tier')} tier)")
                print(f"      Fees:     Delivery ${q['delivery_fee']:.2f} | Waiver ${q['damage_waiver']:.2f} | Tax ${q['tax']:.2f}")

            elif decision == "REQUEST_INFO" and result.get("missing_info"):
                print("   Missing Requirements Needed:")
                for f in result["missing_info"].get("missing_fields", []):
                    print(f"      • {f}")

            elif decision == "MANUAL_REVIEW" and result.get("review_ticket"):
                t = result["review_ticket"]
                print(f"   🚨 Operations Review Ticket (Priority: {t.get('priority')}):")
                print(f"      Next Step: {t.get('recommendation')}")
                for tr in t.get("triggers", []):
                    print(f"      • {tr}")

            if result.get("reasoning"):
                print(f"\n🧠 Agent Reasoning:\n{result.get('reasoning')}")

        print("-" * 70)

        # Append to history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": result.get("message", f"Decision: {decision}")})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    if "--test" in sys.argv:
        run_automated_tests()
    else:
        interactive_chat()

