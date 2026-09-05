# Equipment Rental Decision Agent

### Problem #5 -- Full Build Plan for the Cymonic Agentic Workforce Hackathon

_Round 2 Prep Notes -- September 2026_

---

## Executive Snapshot

Problem #5, the **Equipment Rental Decision Agent**, asks for something narrower and sharper than it first appears: not "build a chatbot for contractors," but build a system that can be trusted to make a judgment call -- auto-quote, ask for more information, or escalate to a human -- on contracts that can be worth thousands of dollars, in a domain where the two failure directions (moving too slow and losing the deal, or moving too fast and creating liability) are both expensive.

That framing is exactly what makes it a strong choice for this hackathon. It rewards genuine decision logic over a pretty interface, it has a natural three-way branch that maps cleanly onto a scorecard, and it produces concrete, inspectable artifacts (a quote breakdown, a qualification scorecard, a review ticket) that are easy to demo convincingly in a five-minute walkthrough.

This document works through the problem end to end: the architecture, the dataset your team should build, the exact scoring and decision logic (with worked numbers, not just a description), the LLM prompts, the UI layout, a file structure you can start coding against immediately, an hour-by-hour plan for a four-person team, a demo script, and a copy-paste starter dataset in the appendix.

## Reframing the Problem

Strip away the hackathon language and the business problem is a dispatcher's dilemma. A rental yard gets an inquiry -- by phone, email, or web form, often outside business hours -- for a piece of equipment that might be worth $150 a day or $1,800 a day. Three things can go wrong, and the agent's whole job is to avoid all three at once:

- **Respond too slowly, or push everything to a human.** Contractors who need equipment tomorrow morning will call the next supplier on the list. Every inquiry that sits in a queue for "someone to review Monday" is a probable lost sale.
- **Auto-quote indiscriminately.** Confirming a rental before checking real availability creates double-bookings. Confirming a rental to a customer with a track record of damage, or on equipment that legally requires a license the customer hasn't verified, creates liability the company wears, not the agent.
- **Ask everyone the same clarifying questions.** A trusted, ten-time repeat customer renting a generator does not need the same friction as a brand-new company asking to rent a crane. Uniform friction annoys good customers as much as uniform trust exposes the business to bad ones.

The agent's value is in **routing each inquiry to the right amount of friction** -- full trust and an instant number for the easy cases, a short clarifying exchange for the ambiguous cases, and a flagged, reasoned escalation for the genuinely risky cases. That is the design target for every section that follows.

## System Architecture

The system is a linear pipeline with one branch point. Each stage has a single, testable responsibility, which matters both for code quality and for being able to explain the system clearly to judges.

```
Raw Inquiry (free text, form, or voice-to-text transcript)
        |
        v
 [1] PARSER            (LLM call)        -> structured_inquiry.json
        |
        v
 [2] VERIFIER          (deterministic)   -> availability_ok / license_ok / site_ok
        |
        v
 [3] SCORER            (weighted rubric) -> five-factor scorecard, 0-100
        |
        v
 [4] HARD GATES        (policy rules)    -> override flags, if any
        |
        v
 [5] DECISION ROUTER   (thresholds)      -> AUTO-QUOTE | REQUEST INFO | MANUAL REVIEW
        |
        v
 [6] ACTION GENERATOR  (deterministic)   -> quote / info-request draft / review ticket
        |
        v
 [7] REASONING WRITER  (LLM call)        -> human-readable explanation, grounded in [3]-[6]
        |
        v
 [8] PERSISTENCE       -> inquiry_log.json updated with decision, scorecard, action, timestamp
        |
        v
      STREAMLIT UI renders every stage above for the live demo
```

The design choice worth calling out explicitly, because it is the difference between a system that looks like "an LLM guessed" and one that looks like real business logic: **the LLM never makes the auto-quote / info / review decision directly.** It only does two things -- turn messy text into structured fields (stage 1), and turn a computed scorecard into readable prose (stage 7). Stages 3-6, the actual judgment, are plain Python running against numbers and rules you can print, log, and defend. This is also what lets you demonstrate "dynamic reasoning, not hardcoded logic" convincingly: you can toggle one input (say, a license-verified flag) live in the demo and show the score and decision change in response, which is something a hardcoded if/else chain over inputs can also technically do -- so what actually proves _dynamic_ reasoning is that the decision comes from a transparent weighted computation over several interacting factors plus policy overrides, not a single lookup.

## Dataset Design

No dataset is provided, so the dataset you design **is** part of the submission. Build four linked JSON files. Realism comes from including fields that do not matter for the happy path but matter for edge cases -- damage history, hazard classification, policy flags -- because those are exactly the fields your scoring engine will reason over.

### Equipment Inventory

Each record needs pricing, a hazard/liability profile, licensing requirements, and a live availability calendar so the verifier has something real to check against.

Suggested fields: `id`, `name`, `category`, `daily_rate`, `weekly_rate`, `replacement_value`, `hazard_class` (Low / Medium / High), `license_required`, `site_requirements`, `booked_dates` (list of date ranges), `policy_flag` (e.g. `never_auto_quote` for high-liability classes), `maintenance_status`.

### Contractor Records

This is what lets "customer trust" be computed rather than assumed. Suggested fields: `id`, `company_name`, `rental_count`, `on_time_payment_rate`, `damage_incidents` (count, with a short note per incident), `license_on_file` (list of verified certifications), `tier` (Gold / Silver / Unrated / Flagged), `company_age_months`.

### Policy Configuration

This is the file that encodes "what the business will never auto-approve regardless of score" -- it is what makes the hard-gate layer feel like real underwriting rather than an arbitrary threshold. Suggested fields: `never_auto_quote_categories`, `new_customer_contract_ceiling`, `rush_window_hours`, `rush_surcharge_pct`, `loyalty_discount_pct_by_tier`, `damage_waiver_pct`, `tax_pct`, `default_delivery_fee`.

### Inquiry Log

This is the table your system writes to, and it is what you show growing live during the demo. Suggested fields: `inquiry_id`, `timestamp`, `raw_text`, `parsed_fields`, `contractor_id`, `equipment_id`, `scorecard`, `hard_gates_triggered`, `decision`, `action_payload`, `status` (Open / Quoted / Awaiting Info / Under Review).

A ready-to-use starter version of all four files, with realistic sample records, is in the Appendix -- your team can paste it in and start wiring logic against real data within the first thirty minutes instead of debating field names.

## The Reasoning Engine

This is the core of the submission. Each step below is deliberately narrow so it can be unit-tested on its own before you wire the pipeline together.

### Step 1 -- Parsing (LLM call)

Contractor inquiries arrive as messy free text -- an email, a transcribed voicemail, a web form comment box. An LLM call turns this into structured fields far more robustly than regex, especially for phrases like "need it by tomorrow morning" or "not 100% sure on dates yet."

Suggested system prompt:

```
You are a parsing assistant for a heavy equipment rental company.
Extract structured fields from the raw contractor inquiry below.
Return ONLY a valid JSON object with these keys:
equipment_requested, duration_value, duration_unit, start_date,
end_date, site_location, site_access_notes, license_mentioned,
urgency_level ("standard" or "rush"), contractor_name_mentioned,
raw_intent_summary.
If a field cannot be determined from the text, set it to null.
Do not include any text outside the JSON object.
```

Treat every `null` field as a real signal, not noise -- it flows directly into the completeness score in Step 3 and, later, into what the "request more info" draft actually asks for.

### Step 2 -- Verification (deterministic)

Plain Python, no LLM needed, and it should be written as pure functions so you can unit test them independently:

- **Availability check** -- do the requested dates overlap any range in the equipment's `booked_dates`? If yes, is a substitute unit of the same category free for the same window?
- **License check** -- does `license_required` for the equipment appear in the contractor's `license_on_file` list? If the inquiry did not mention a license at all, this is `unverified`, not `false` -- that distinction matters for whether you request info or flag for review.
- **Site check** -- does `site_access_notes` from the parsed inquiry satisfy `site_requirements` on the equipment record, at least well enough to not need a survey?

### Step 3 -- Scoring (weighted rubric)

A single weighted score across five factors, each independently computed and independently explainable -- this is what a judge can poke at and get a sensible answer for.

| Factor                 | Max Points | Scores high when...                                       | Scores low when...                                |
| ---------------------- | ---------- | --------------------------------------------------------- | ------------------------------------------------- |
| Availability & Fit     | 30         | Exact equipment is free for the full requested window     | Booked, or only a partial/substitute match exists |
| Criteria Completeness  | 20         | Dates, license, site access, and logistics are all stated | Any of the four is missing or ambiguous           |
| Customer Trust         | 25         | Repeat customer, clean payment and safety history         | New/unverifiable customer, or prior incidents     |
| Liability & Value Risk | 15         | Low replacement value, low hazard class                   | High-value or high-hazard machinery               |
| Timing Context         | 10         | Standard lead time, inquiry during business hours         | After-hours or under 24-hour rush request         |

Total: 100 points. Keep the weights in a single config dictionary (not scattered across functions) so you can tune them live if a judge asks "what if trust mattered more than completeness?" -- being able to answer that question by changing one number is a strong signal of a well-factored system.

### Step 4 -- Hard Gates (policy overrides)

This layer is what separates a toy scoring demo from something that reads as genuine business judgment, and it is the detail most teams will skip under time pressure -- which is exactly why including it stands out. A hard gate can force a stricter outcome **regardless of the numeric score**:

- Equipment is on the `never_auto_quote_categories` policy list (e.g., large mobile cranes, demolition equipment) -- always at least Manual Review.
- License is required for the equipment and remains unverified -- never Auto-Quote.
- The contractor has one or more prior damage or liability incidents on file -- forces Manual Review.
- Two overlapping bookings would result from proceeding (a would-be double-booking) -- forces at least Request More Info.
- A brand-new customer (zero rental history) is requesting a contract above `new_customer_contract_ceiling` -- forces Manual Review even with a clean-looking score.

### Step 5 -- Decision Mapping

| Score Range   | Decision (if no hard gate fires)                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| 75 -- 100     | Auto-Quote                                                                                             |
| 45 -- 74      | Request More Info                                                                                      |
| 0 -- 44       | Manual Review                                                                                          |
| _(any score)_ | **Manual Review or Request More Info if a hard gate fires** -- gate outcome always wins over the score |

### Step 6 -- Action Generation (deterministic)

**If Auto-Quote**, compute a real breakdown:

```
Subtotal = daily_rate x duration_days
           (use weekly_rate x ceil(duration/7) instead if duration >= 7 days)

+ Delivery / Mobilization Fee   (flat, from equipment record)
+ Damage Waiver                 (damage_waiver_pct of Subtotal)
+ Rush Surcharge                (rush_surcharge_pct of Subtotal, only if lead time < rush_window_hours)
- Loyalty Discount              (tier discount pct of Subtotal, only for Gold/Silver tiers)
= Pre-Tax Total
+ Tax                           (tax_pct of Pre-Tax Total)
= Final Quote
```

**If Request More Info**, generate a short, specific draft naming exactly the missing fields from Step 1/2 -- never a generic "please provide more details."

**If Manual Review**, generate a review ticket: priority level (Urgent if `urgency_level = rush`), the specific hard gates or low-scoring factors that triggered it, and a one-line recommended next step for the human reviewer (e.g., "recommend a site-access survey before quoting").

### Step 7 -- Reasoning Writer (LLM call)

A short prompt that turns the scorecard and decision into a readable paragraph, explicitly grounded so it cannot invent facts:

```
You are drafting an internal decision note for a rental operations team.
Given the scorecard and decision below, write 3-4 sentences a human
manager would find clear and defensible. Reference only the factors
present in the data. Do not invent facts not present in the input.

Scorecard: {scorecard_json}
Hard gates triggered: {hard_gates_list}
Decision: {decision}
```

### Step 8 -- Persistence

Append the full record -- raw text, parsed fields, scorecard, gates, decision, and action payload -- to `inquiry_log.json`. This is what makes the "updated dataset" requirement visible and is worth surfacing directly in the UI, not just writing silently to disk.

## Worked Examples

Three full walkthroughs, one per decision branch, using the sample dataset in the Appendix. Numbers are computed exactly as the formulas above define them, so your team can use these as unit test fixtures.

### Example A -- Auto-Quote

**Inquiry (from Ferreira Builders LLC, Gold tier, contractor C-201):**
"Need the mini excavator for 5 days starting Sept 15, site is our usual compacted lot, our operator's HEO cert is already on file. Can you confirm and send pricing?"

**Parsed:** equipment = mini excavator (EQ-1001); duration = 5 days; site access = compacted lot (matches requirement); license = mentioned as on file.

**Verification:** EQ-1001 is open for the requested window; license required is a Standard Heavy Equipment Operator cert, and C-201's file shows it verified; site access matches.

**Scorecard:**

| Factor                               | Score         |
| ------------------------------------ | ------------- |
| Availability & Fit                   | 30 / 30       |
| Criteria Completeness                | 20 / 20       |
| Customer Trust (Gold, clean history) | 25 / 25       |
| Liability & Value Risk (Low hazard)  | 15 / 15       |
| Timing Context (6-day lead time)     | 10 / 10       |
| **Total**                            | **100 / 100** |

No hard gates triggered. **Decision: Auto-Quote.**

**Quote breakdown:** Subtotal = $220 x 5 = $1,100. Delivery fee $75. Damage waiver (5%) = $55. No rush surcharge (lead time is 6 days). Gold loyalty discount (5%) = -$55. Pre-tax total = $1,175. Tax (8%) = $94. **Final quote: $1,269.**

### Example B -- Request More Info

**Inquiry (from Nomad Site Services, Silver tier, contractor C-202):**
"Need a scissor lift for a job next week, maybe 4-5 days, not sure on exact dates yet, will confirm site details soon."

**Parsed:** equipment = scissor lift (EQ-1004); duration = approx. 4-5 days; start date = unresolved; site access = not specified; license = not mentioned.

**Verification:** EQ-1004 has an existing booking Sept 6-8, but without exact requested dates availability cannot be confirmed either way. The required Aerial Work Platform certification is not confirmed on C-202's file.

**Scorecard:**

| Factor                                                         | Score        |
| -------------------------------------------------------------- | ------------ |
| Availability & Fit (dates unresolved)                          | 15 / 30      |
| Criteria Completeness (dates, license, site all missing/vague) | 3 / 20       |
| Customer Trust (Silver, one late payment)                      | 15 / 25      |
| Liability & Value Risk (Medium hazard)                         | 8 / 15       |
| Timing Context (standard lead time)                            | 10 / 10      |
| **Total**                                                      | **51 / 100** |

Falls in the 45-74 band; license status is unverified rather than confirmed-false, so this is a completeness gap, not a hard-gate trust problem. **Decision: Request More Info.**

**Generated draft:** "Thanks for reaching out -- to lock in availability and pricing we need: (1) your exact start and end dates, (2) confirmation of your Aerial Work Platform certification on file, and (3) a quick note on the site surface. Once we have these we can issue your quote right away."

### Example C -- Manual Review

**Inquiry (from Titan Demolition Inc., Flagged tier, contractor C-204), received 11:40 PM:**
"Need the 50-ton mobile crane for a demolition job starting tomorrow morning, 3 days, downtown site with tight access. Big contract riding on this, can you rush the quote?"

**Parsed:** equipment = 50-ton mobile crane (EQ-1002); duration = 3 days; start = next day (rush, under 24-hour lead time); site access = "tight downtown access," flagged as needing a survey.

**Verification:** crane certification is on file and verified for C-204, but the crane category itself is on the `never_auto_quote_categories` policy list, and C-204's record shows a prior boom-damage incident.

**Scorecard (for reference -- the gates decide the outcome here, not the score):**

| Factor                                               | Score        |
| ---------------------------------------------------- | ------------ |
| Availability & Fit                                   | 30 / 30      |
| Criteria Completeness (site survey still needed)     | 15 / 20      |
| Customer Trust (prior damage incident caps this low) | 5 / 25       |
| Liability & Value Risk (High hazard, high value)     | 0 / 15       |
| Timing Context (after-hours rush)                    | 5 / 10       |
| **Total**                                            | **55 / 100** |

On score alone this would land in "Request More Info" -- but **two hard gates fire**: the equipment category is never-auto-quote, and the contractor has a prior liability incident. Gates override the score. **Decision: Manual Review, flagged Urgent.**

**Generated review ticket:** "Urgent -- respond within 2 hours. Crane category requires manual approval per policy regardless of score. Contractor has one prior damage incident (boom, 2025). Site access described as tight with no survey on file. Recommend a site-access survey before any commitment is made."

These three examples are worth rehearsing verbatim for the demo -- they are the fastest way to prove the system reasons rather than templates.

## UI/UX Design

Streamlit is the right tool here: fast to build, no styling investment needed, and its native widgets (tables, metrics, forms) map directly onto what needs to be shown. Four tabs cover everything:

1. **New Inquiry** -- a text area for pasting the raw inquiry (plus a couple of quick-pick sample inquiries as buttons, so the demo doesn't depend on typing live), and an "Analyze" button that runs the full pipeline.
2. **Decision Output** -- a color-coded badge (green / yellow / red) for the decision, the reasoning paragraph, a bar chart of the five scorecard factors (`st.bar_chart` is enough, no need for anything fancier), and the action payload rendered appropriately: a quote table, an info-request draft in a text box, or a review ticket card.
3. **Inquiry Log** -- a live table (`st.dataframe`) of every processed inquiry with columns for contractor, equipment, decision, and status, filterable by decision type. This is where you visibly show the dataset "updating."
4. **Dataset Viewer** -- a read-only view of the equipment and contractor tables, so judges can see the "database" the agent is reasoning against rather than taking it on faith.

An optional fifth tab, **Analytics** (percentage auto-quoted, average score, total value quoted today), is a nice stretch goal if time allows in the last hour, but is not essential to a passing, well-argued demo.

## Tech Stack & Project Structure

```
equipment-rental-agent/
|-- app.py                    Streamlit UI entrypoint
|-- data/
|   |-- equipment.json
|   |-- contractors.json
|   |-- policy_config.json
|   `-- inquiry_log.json      grows at runtime
|-- engine/
|   |-- parser.py             LLM-based inquiry parsing (Step 1)
|   |-- verifier.py           availability / license / site checks (Step 2)
|   |-- scorer.py             weighted scoring engine (Step 3)
|   |-- decision.py           hard gates + threshold mapping (Steps 4-5)
|   |-- actions.py            quote / info-request / review-ticket generators (Step 6)
|   `-- reasoning.py          LLM-based explanation writer (Step 7)
|-- utils/
|   `-- storage.py            read/write the JSON "database" (Step 8)
|-- requirements.txt
`-- .env                      API key, not committed
```

Core libraries: `streamlit`, the Anthropic Python SDK (or `requests` against the API directly), `pandas` for the log table, `python-dateutil` for date-range overlap checks, and plain `json` for storage -- no database server needed for a five-hour build.

## Build Plan: Hour-by-Hour (Team of 4)

Suggested roles: **A** -- Data & Domain (datasets, policy rules); **B** -- Backend/Decision Engine (verifier, scorer, decision, actions); **C** -- LLM Integration (parser, reasoning prompts, API wiring); **D** -- Frontend/Streamlit + Demo prep.

| Time      | A -- Data & Domain                                                                                     | B -- Decision Engine                                                                           | C -- LLM Integration                                               | D -- Frontend & Demo                                      |
| --------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------- |
| 0:00-0:30 | Whole team: agree on scoring weights, hard gates, and decision thresholds together before splitting up |                                                                                                |                                                                    |                                                           |
| 0:30-1:30 | Build equipment/contractor/policy JSON with realistic fields                                           | Scaffold `verifier.py` and `decision.py` with rubric constants                                 | Build API wrapper; test the parsing prompt on 2-3 sample inquiries | Scaffold Streamlit tabs and layout with mock data         |
| 1:30-2:30 | Write 15-20 seed inquiries covering edge cases (rush, vague, flagged customer)                         | Implement `scorer.py` + `decision.py` fully; unit-test against the three worked examples above | Wire `reasoning.py`; tune tone/length of the explanation output    | Connect "New Inquiry" tab to real backend functions       |
| 2:30-3:30 | Support integration testing with real data                                                             | Full pipeline integration: parser to verifier to scorer to decision to actions to reasoning    | Same                                                               | Build Inquiry Log tab and scorecard bar chart             |
| 3:30-4:15 |                                                                                                        | Bug fixes; handle malformed/partial LLM parses gracefully                                      | Add a fallback path if the API call fails mid-demo                 | Add color-coded decision badges; build Dataset Viewer tab |
| 4:15-4:45 | Whole team: rehearse the demo script against the three worked examples                                 |                                                                                                |                                                                    |                                                           |
| 4:45-5:00 | Buffer, final polish, submission                                                                       |                                                                                                |                                                                    |                                                           |

## Demo Script (about 4 minutes)

1. **(20s)** State the problem in one breath: auto-quote everything and you erode margin and take on liability; manual-review everything and you lose fast-moving deals. The agent's job is deciding which inquiries need which.
2. **(20s)** Open the Dataset Viewer briefly -- "here's our fleet and our contractor history, including a flagged account and a Gold-tier repeat customer."
3. **(80s)** Submit Example A live (Ferreira, mini excavator) -- walk through the scorecard, then the full quote breakdown, narrating that every line item traces back to a real field in the dataset.
4. **(60s)** Submit Example B live (Nomad, vague scissor-lift request) -- show the score landing in the middle band and the specific, non-generic info request it generates.
5. **(60s)** Submit Example C live (Titan, after-hours crane request from a flagged account) -- this is the moment to say out loud: "the score alone would have said 'ask for more info,' but two policy gates override it, because company policy says crane-class equipment and flagged accounts always go to a human." This line is what proves the reasoning is layered, not a single lookup.
6. **(20s)** Flip to the Inquiry Log -- all three now sit there with their decisions and statuses, visibly updated.
7. **(20s)** Close on the business impact in one line: fast, defensible answers for the easy 70% of inquiries, and protected margin and reduced liability exposure on the risky ones.

## Judging Criteria Alignment

| What judges look for                   | Where this design delivers it                                                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Dynamic reasoning, not hardcoded logic | Weighted five-factor scorer plus a separate policy-gate layer; changing one input changes the score and can flip the decision |
| Realistic, extended dataset            | Contractor tiers, hazard classes, damage-incident history, and policy flags -- fields that actually get used, not decoration  |
| Business-first framing                 | The quote math mirrors real rental-industry line items (mobilization fee, damage waiver, rush surcharge, loyalty discount)    |
| Clean, functional UI                   | Four focused Streamlit tabs; no time spent on custom styling                                                                  |
| A working, explainable end-to-end demo | Three rehearsed examples that hit all three decision branches, each traceable back to specific data fields                    |

## Pitfalls to Avoid

- **Don't let the LLM make the final call.** If the auto-quote/info/review decision itself comes from an LLM prompt, it will look unpredictable the moment a judge asks "what if you change one word in the inquiry?" Keep the decision in deterministic code; use the LLM only to parse input and phrase output.
- **Don't skip the hard-gate layer to save time.** A purely score-based system falls apart under one obvious judge question: "what if a risky customer happens to score well?" The gate layer is a small amount of code for a large amount of credibility.
- **Don't over-invest in UI polish.** Four working Streamlit tabs beat one beautifully styled tab with broken logic behind it.
- **Don't forget to make persistence visible.** Judges may ask directly whether the dataset updates -- have the Inquiry Log tab open and show a new row appear live.
- **Have an offline fallback for the LLM calls.** If the API is slow or unavailable during judging, pre-cache the outputs for your three demo examples so the walkthrough never stalls.

## Appendix: Starter Dataset (copy-paste ready)

### `equipment.json`

```json
[
  {
    "id": "EQ-1001",
    "name": "Mini Excavator (3-ton)",
    "category": "Earthmoving",
    "daily_rate": 220,
    "weekly_rate": 1200,
    "replacement_value": 45000,
    "hazard_class": "Low",
    "license_required": "Standard Heavy Equipment Operator Cert",
    "site_requirements": "Firm or compacted ground, 8ft clearance",
    "booked_dates": [],
    "policy_flag": null,
    "maintenance_status": "OK"
  },
  {
    "id": "EQ-1002",
    "name": "Mobile Crane (50-ton)",
    "category": "Lifting",
    "daily_rate": 1800,
    "weekly_rate": 9800,
    "replacement_value": 650000,
    "hazard_class": "High",
    "license_required": "NCCCO Crane Operator + Rigger",
    "site_requirements": "Load-bearing surface, overhead clearance survey",
    "booked_dates": [],
    "policy_flag": "never_auto_quote",
    "maintenance_status": "OK"
  },
  {
    "id": "EQ-1003",
    "name": "Towable Diesel Generator (100kW)",
    "category": "Power",
    "daily_rate": 150,
    "weekly_rate": 800,
    "replacement_value": 22000,
    "hazard_class": "Low",
    "license_required": null,
    "site_requirements": "Standard access",
    "booked_dates": [],
    "policy_flag": null,
    "maintenance_status": "OK"
  },
  {
    "id": "EQ-1004",
    "name": "Scissor Lift (26ft)",
    "category": "Aerial",
    "daily_rate": 130,
    "weekly_rate": 650,
    "replacement_value": 18000,
    "hazard_class": "Medium",
    "license_required": "Aerial Work Platform Cert",
    "site_requirements": "Flat surface, indoor or outdoor",
    "booked_dates": [["2026-09-06", "2026-09-08"]],
    "policy_flag": null,
    "maintenance_status": "OK"
  },
  {
    "id": "EQ-1005",
    "name": "Concrete Mixer Truck",
    "category": "Concrete",
    "daily_rate": 400,
    "weekly_rate": 2200,
    "replacement_value": 95000,
    "hazard_class": "Medium",
    "license_required": "Commercial Driver's License",
    "site_requirements": "Road access for truck",
    "booked_dates": [],
    "policy_flag": null,
    "maintenance_status": "OK"
  },
  {
    "id": "EQ-1006",
    "name": "Skid Steer Loader",
    "category": "Earthmoving",
    "daily_rate": 180,
    "weekly_rate": 950,
    "replacement_value": 38000,
    "hazard_class": "Low",
    "license_required": "Standard Operator Cert",
    "site_requirements": "Standard access",
    "booked_dates": [],
    "policy_flag": null,
    "maintenance_status": "OK"
  }
]
```

### `contractors.json`

```json
[
  {
    "id": "C-201",
    "company_name": "Ferreira Builders LLC",
    "rental_count": 14,
    "on_time_payment_rate": 1.0,
    "damage_incidents": [],
    "license_on_file": ["Standard Heavy Equipment Operator Cert"],
    "tier": "Gold",
    "company_age_months": 60
  },
  {
    "id": "C-202",
    "company_name": "Nomad Site Services",
    "rental_count": 3,
    "on_time_payment_rate": 0.85,
    "damage_incidents": [],
    "license_on_file": ["Standard Operator Cert"],
    "tier": "Silver",
    "company_age_months": 18
  },
  {
    "id": "C-203",
    "company_name": "Quick Build Co.",
    "rental_count": 0,
    "on_time_payment_rate": null,
    "damage_incidents": [],
    "license_on_file": [],
    "tier": "Unrated",
    "company_age_months": 2
  },
  {
    "id": "C-204",
    "company_name": "Titan Demolition Inc.",
    "rental_count": 6,
    "on_time_payment_rate": 0.9,
    "damage_incidents": [
      {
        "date": "2025-04-02",
        "note": "Crane boom contact, minor structural damage"
      }
    ],
    "license_on_file": ["NCCCO Crane Operator + Rigger"],
    "tier": "Flagged",
    "company_age_months": 40
  }
]
```

### `policy_config.json`

```json
{
  "never_auto_quote_categories": ["Lifting", "Demolition"],
  "new_customer_contract_ceiling": 10000,
  "rush_window_hours": 24,
  "rush_surcharge_pct": 0.1,
  "loyalty_discount_pct_by_tier": {
    "Gold": 0.05,
    "Silver": 0.02,
    "Unrated": 0.0,
    "Flagged": 0.0
  },
  "damage_waiver_pct": 0.05,
  "tax_pct": 0.08,
  "default_delivery_fee": 75
}
```

### Sample seed inquiries (for `inquiry_log.json` testing)

```json
[
  {
    "contractor_id": "C-201",
    "raw_text": "Need the mini excavator for 5 days starting Sept 15, site is our usual compacted lot, our operator's HEO cert is already on file. Can you confirm and send pricing?"
  },
  {
    "contractor_id": "C-202",
    "raw_text": "Need a scissor lift for a job next week, maybe 4-5 days, not sure on exact dates yet, will confirm site details soon."
  },
  {
    "contractor_id": "C-204",
    "raw_text": "Need the 50-ton mobile crane for a demolition job starting tomorrow morning, 3 days, downtown site with tight access. Big contract riding on this, can you rush the quote?"
  },
  {
    "contractor_id": "C-203",
    "raw_text": "Hi, first time renting from you -- interested in the concrete mixer truck for about 2 weeks starting next month for a big commercial pour."
  }
]
```
