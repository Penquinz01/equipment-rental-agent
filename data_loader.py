"""
data_loader.py - Data Loading and Audit Decision Logging Helpers
Person D Responsibility: Data loading helpers & Decision CSV logging
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from utils import (
    BASE_DIR,
    DATA_DIR,
    DECISION_LOG_PATH,
    generate_inquiry_id,
    get_timestamp,
    safe_get,
    summarize_reasoning,
)

EQUIPMENT_CSV_PATH = DATA_DIR / "equipment.csv"
CONTRACTORS_CSV_PATH = DATA_DIR / "contractors.csv"
INQUIRIES_CSV_PATH = DATA_DIR / "inquiries.csv"

DECISION_LOG_COLUMNS = [
    "timestamp",
    "inquiry_id",
    "contractor_name",
    "equipment_name",
    "score",
    "decision",
    "total_quote",
    "reasoning_summary",
]


def safe_load_csv(file_path: Path, expected_columns: Optional[list] = None) -> pd.DataFrame:
    """Helper to safely load a CSV file, returning an empty DataFrame if missing or corrupt."""
    if not file_path.exists():
        return pd.DataFrame(columns=expected_columns or [])
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception:
        return pd.DataFrame(columns=expected_columns or [])


def load_equipment_df() -> pd.DataFrame:
    """Load equipment catalog dataset safely."""
    return safe_load_csv(EQUIPMENT_CSV_PATH)


def load_contractors_df() -> pd.DataFrame:
    """Load contractors directory dataset safely."""
    return safe_load_csv(CONTRACTORS_CSV_PATH)


def load_inquiries_df() -> pd.DataFrame:
    """Load past inquiries dataset safely."""
    return safe_load_csv(INQUIRIES_CSV_PATH)


def load_decision_log_df() -> pd.DataFrame:
    """Load decision log audit dataset safely."""
    return safe_load_csv(DECISION_LOG_PATH, DECISION_LOG_COLUMNS)


def log_decision(user_text: str, result: Dict[str, Any], log_file_path: Optional[Path] = None) -> None:
    """
    Append a new decision record to data/decision_log.csv.
    NEVER overwrites existing log entries.

    Columns:
    timestamp,inquiry_id,contractor_name,equipment_name,score,decision,total_quote,reasoning_summary
    """
    target_path = log_file_path or DECISION_LOG_PATH

    # Ensure data directory exists safely
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(result, dict):
        result = {}

    timestamp = get_timestamp()
    inquiry_id = generate_inquiry_id(target_path)

    # Safe extraction of contractor & equipment name (blank string if None or missing)
    contractor_name = safe_get(result, "contractor_name")
    contractor_str = "" if contractor_name is None else str(contractor_name).strip()

    equipment_name = safe_get(result, "equipment_name")
    equipment_str = "" if equipment_name is None else str(equipment_name).strip()

    # Score extraction from scorecard.total
    score_val = safe_get(result, "scorecard", "total")
    if score_val is None:
        score_val = safe_get(result, "scorecard", "total_score")
    score_str = "" if score_val is None else str(score_val).strip()

    # Decision string extraction
    decision = str(safe_get(result, "decision", default="UNRECOGNIZED")).strip()

    # Total quote extraction (only for AUTO_QUOTE)
    total_quote = ""
    if decision == "AUTO_QUOTE":
        quote_total = safe_get(result, "quote", "final_total")
        if quote_total is not None:
            try:
                total_quote = f"{float(quote_total):.2f}"
            except (ValueError, TypeError):
                total_quote = str(quote_total).strip()

    # Compact reasoning summary
    reasoning_summary = summarize_reasoning(result)

    row_data = {
        "timestamp": timestamp,
        "inquiry_id": inquiry_id,
        "contractor_name": contractor_str,
        "equipment_name": equipment_str,
        "score": score_str,
        "decision": decision,
        "total_quote": total_quote,
        "reasoning_summary": reasoning_summary,
    }

    new_row_df = pd.DataFrame([row_data], columns=DECISION_LOG_COLUMNS)

    # Write header only if file does not exist or is 0 bytes
    file_exists = target_path.exists() and target_path.stat().st_size > 0

    new_row_df.to_csv(
        target_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )
