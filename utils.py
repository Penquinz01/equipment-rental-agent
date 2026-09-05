"""
utils.py - Reusable Utility Helpers for Equipment Rental Decision Agent
Person D Responsibility: Integration support, utility functions, audit log exporting.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Project base directory resolution relative to this file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DECISION_LOG_PATH = DATA_DIR / "decision_log.csv"


def get_timestamp() -> str:
    """Return a readable local timestamp string (e.g., '2026-09-05 11:32:41')."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_get(d: Any, *keys: Union[str, int], default: Any = None) -> Any:
    """
    Safely retrieve nested dictionary/list values without raising KeyError, TypeError, or IndexError.

    Example:
        safe_get(result, "scorecard", "total")
    """
    curr = d
    for key in keys:
        if isinstance(curr, dict):
            curr = curr.get(key, None)
        elif isinstance(curr, (list, tuple)) and isinstance(key, int):
            if 0 <= key < len(curr):
                curr = curr[key]
            else:
                return default
        else:
            return default
        if curr is None:
            return default
    return curr


def generate_inquiry_id(log_file_path: Optional[Path] = None) -> str:
    """
    Generate the next sequential request ID based on existing decision_log.csv.
    Examples: REQ001, REQ002, REQ003.
    If the log does not exist or is empty, returns REQ001.
    Prevents IDs from resetting across application restarts.
    """
    target_path = log_file_path or DECISION_LOG_PATH
    if not target_path.exists():
        return "REQ001"

    try:
        df = pd.read_csv(target_path, dtype=str)
        if df.empty or "inquiry_id" not in df.columns:
            return "REQ001"

        ids = df["inquiry_id"].dropna().tolist()
        max_num = 0
        for item in ids:
            match = re.search(r"REQ(\d+)", str(item), re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

        next_num = max_num + 1
        return f"REQ{next_num:03d}"
    except Exception:
        return "REQ001"


def summarize_reasoning(result: Optional[Dict[str, Any]]) -> str:
    """
    Convert reasoning/scorecard from agent result into a compact one-line summary.
    Maximum ~300 characters.
    Prefer generating summary from scorecard factors if available.
    """
    if not isinstance(result, dict):
        return "No reasoning provided."

    scorecard = safe_get(result, "scorecard")
    decision = safe_get(result, "decision", default="UNKNOWN")

    # If scorecard dictionary is available
    if isinstance(scorecard, dict):
        parts: List[str] = []
        # Check factors in Person B's scorecard
        factor_keys = ["availability", "completeness", "customer_trust", "contractor_trust", "liability_risk", "timing"]
        for fk in factor_keys:
            factor = scorecard.get(fk)
            if isinstance(factor, dict) and "score" in factor and "max" in factor:
                display_name = fk.replace("_", " ")
                parts.append(f"{display_name} {factor['score']}/{factor['max']}")

        total = safe_get(scorecard, "total")
        max_total = safe_get(scorecard, "max_total", default=100)

        if parts:
            summary = "; ".join(parts)
            if total is not None:
                summary += f". Final score {total}/{max_total} → {decision}."
            else:
                summary += f". Decision → {decision}."
            if len(summary) <= 300:
                return summary

        if total is not None:
            return f"Score {total}/{max_total} → {decision}."

    # Fallback to result["reasoning"] string if provided
    raw_reasoning = safe_get(result, "reasoning")
    if isinstance(raw_reasoning, str) and raw_reasoning.strip():
        # Collapse excessive newlines and whitespace
        clean_text = re.sub(r"\s+", " ", raw_reasoning).strip()
        # Remove private thought headers if present
        clean_text = clean_text.replace("🧠 Agent Thought Process:", "").strip()
        if len(clean_text) > 300:
            clean_text = clean_text[:297] + "..."
        return clean_text

    # Decision payload fallbacks
    missing_msg = safe_get(result, "missing_info", "message")
    if missing_msg:
        return f"Missing Info: {missing_msg}"[:300]

    unrec_msg = safe_get(result, "message")
    if unrec_msg:
        return f"Message: {unrec_msg}"[:300]

    rec_msg = safe_get(result, "review_ticket", "recommendation")
    if rec_msg:
        return f"Review Ticket: {rec_msg}"[:300]

    return f"Decision processed as {decision}."


def export_log_as_text(log_file_path: Optional[Path] = None) -> str:
    """
    Read data/decision_log.csv and return a human-readable plain-text audit log.
    If file does not exist or is empty, returns 'No rental audit records available.'
    """
    target_path = log_file_path or DECISION_LOG_PATH
    if not target_path.exists():
        return "No rental audit records available."

    try:
        df = pd.read_csv(target_path, dtype=str).fillna("")
        if df.empty:
            return "No rental audit records available."

        lines = [
            "==================================================",
            "EQUIPMENT RENTAL AGENT AUDIT LOG",
            "=================================================="
        ]

        valid_entries = 0
        for _, row in df.iterrows():
            inquiry_id = str(row.get("inquiry_id", "")).strip() or "N/A"
            timestamp = str(row.get("timestamp", "")).strip() or "N/A"
            contractor = str(row.get("contractor_name", "")).strip() or "N/A"
            equipment = str(row.get("equipment_name", "")).strip() or "N/A"
            score_val = str(row.get("score", "")).strip()
            score_str = f"{score_val}/100" if score_val else "N/A"
            decision = str(row.get("decision", "")).strip() or "N/A"

            quote_val = str(row.get("total_quote", "")).strip()
            if quote_val:
                try:
                    quote_str = f"${float(quote_val):.2f}"
                except ValueError:
                    quote_str = f"${quote_val}" if not quote_val.startswith("$") else quote_val
            else:
                quote_str = "N/A"

            reasoning = str(row.get("reasoning_summary", "")).strip() or "N/A"

            lines.append("")
            lines.append(f"Request: {inquiry_id}")
            lines.append(f"Timestamp: {timestamp}")
            lines.append(f"Contractor: {contractor}")
            lines.append(f"Equipment: {equipment}")
            lines.append(f"Score: {score_str}")
            lines.append(f"Decision: {decision}")
            lines.append(f"Total Quote: {quote_str}")
            lines.append(f"Reasoning Summary: {reasoning}")
            lines.append("")
            lines.append("---")
            valid_entries += 1

        if valid_entries == 0:
            return "No rental audit records available."

        return "\n".join(lines)
    except Exception:
        return "No rental audit records available."
