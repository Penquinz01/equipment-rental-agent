"""
test_integration.py - Automated Integration Tests for Person D Modules
Tests decision logging, sequential inquiry ID generation, export text formatting, and defensive handling.
Uses temporary test logs to preserve production data/decision_log.csv.
Does NOT require an LLM API key.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from data_loader import log_decision, DECISION_LOG_COLUMNS
from utils import export_log_as_text, generate_inquiry_id, get_timestamp, safe_get, summarize_reasoning


class TestPersonDIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "test_decision_log.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_1_auto_quote_logging(self):
        result = {
            "decision": "AUTO_QUOTE",
            "contractor_name": "Smith & Sons Construction",
            "equipment_name": "JCB 3CX Backhoe",
            "scorecard": {"total": 92, "max_total": 100},
            "quote": {"final_total": 1458.00},
            "reasoning": "Availability confirmed; required information complete."
        }
        log_decision("Need mini excavator for 5 days", result, log_file_path=self.log_path)

        df = pd.read_csv(self.log_path, dtype=str)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["inquiry_id"], "REQ001")
        self.assertEqual(df.iloc[0]["contractor_name"], "Smith & Sons Construction")
        self.assertEqual(df.iloc[0]["equipment_name"], "JCB 3CX Backhoe")
        self.assertEqual(df.iloc[0]["score"], "92")
        self.assertEqual(df.iloc[0]["decision"], "AUTO_QUOTE")
        self.assertEqual(df.iloc[0]["total_quote"], "1458.00")

    def test_2_request_info_logging(self):
        result = {
            "decision": "REQUEST_INFO",
            "contractor_name": None,
            "equipment_name": "Genie GS-1930 Scissor Lift",
            "scorecard": {"total": 55},
            "missing_info": {"message": "Dates needed"},
            "reasoning": "Missing exact start date."
        }
        log_decision("Need scissor lift next week", result, log_file_path=self.log_path)

        df = pd.read_csv(self.log_path, dtype=str).fillna("")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["decision"], "REQUEST_INFO")
        self.assertEqual(df.iloc[0]["contractor_name"], "")
        self.assertEqual(df.iloc[0]["total_quote"], "")
        self.assertEqual(df.iloc[0]["score"], "55")

    def test_3_manual_review_logging(self):
        result = {
            "decision": "MANUAL_REVIEW",
            "contractor_name": "Titan Demolition Inc",
            "equipment_name": "50-ton Mobile Crane",
            "scorecard": {"total": 45},
            "review_ticket": {"recommendation": "Downtown access safety review needed"},
            "reasoning": "Heavy crane downtown site."
        }
        log_decision("Need 50-ton mobile crane downtown", result, log_file_path=self.log_path)

        df = pd.read_csv(self.log_path, dtype=str).fillna("")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["decision"], "MANUAL_REVIEW")
        self.assertEqual(df.iloc[0]["contractor_name"], "Titan Demolition Inc")
        self.assertEqual(df.iloc[0]["total_quote"], "")

    def test_4_unrecognized_logging(self):
        result = {
            "decision": "UNRECOGNIZED",
            "contractor_name": None,
            "equipment_name": None,
            "scorecard": None,
            "message": "Do not rent lawn mowers.",
            "reasoning": "Outside catalog."
        }
        log_decision("Rent lawn mowers", result, log_file_path=self.log_path)

        df = pd.read_csv(self.log_path, dtype=str).fillna("")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["decision"], "UNRECOGNIZED")
        self.assertEqual(df.iloc[0]["contractor_name"], "")
        self.assertEqual(df.iloc[0]["equipment_name"], "")
        self.assertEqual(df.iloc[0]["score"], "")
        self.assertEqual(df.iloc[0]["total_quote"], "")

    def test_5_sequential_inquiry_ids(self):
        res1 = {"decision": "REQUEST_INFO"}
        res2 = {"decision": "AUTO_QUOTE", "quote": {"final_total": 500}}
        res3 = {"decision": "UNRECOGNIZED"}

        log_decision("test1", res1, log_file_path=self.log_path)
        log_decision("test2", res2, log_file_path=self.log_path)
        log_decision("test3", res3, log_file_path=self.log_path)

        df = pd.read_csv(self.log_path, dtype=str)
        self.assertEqual(list(df["inquiry_id"]), ["REQ001", "REQ002", "REQ003"])

    def test_6_csv_append_behavior(self):
        # First write
        res1 = {"decision": "AUTO_QUOTE", "quote": {"final_total": 100}}
        log_decision("test1", res1, log_file_path=self.log_path)

        df1 = pd.read_csv(self.log_path, dtype=str)
        self.assertEqual(len(df1), 1)

        # Second write (append, preserve existing)
        res2 = {"decision": "MANUAL_REVIEW"}
        log_decision("test2", res2, log_file_path=self.log_path)

        df2 = pd.read_csv(self.log_path, dtype=str)
        self.assertEqual(len(df2), 2)
        self.assertEqual(list(df2.columns), DECISION_LOG_COLUMNS)
        self.assertEqual(df2.iloc[0]["inquiry_id"], "REQ001")
        self.assertEqual(df2.iloc[1]["inquiry_id"], "REQ002")

    def test_7_export_function(self):
        # Missing log test
        missing_log = Path(self.temp_dir.name) / "nonexistent.csv"
        self.assertEqual(export_log_as_text(missing_log), "No rental audit records available.")

        # Populated log export test
        res = {
            "decision": "AUTO_QUOTE",
            "contractor_name": "Smith & Sons Construction",
            "equipment_name": "JCB 3CX Backhoe",
            "scorecard": {"total": 92},
            "quote": {"final_total": 1458.00},
            "reasoning": "Availability confirmed."
        }
        log_decision("Request prompt", res, log_file_path=self.log_path)

        exported_text = export_log_as_text(self.log_path)
        self.assertIn("EQUIPMENT RENTAL AGENT AUDIT LOG", exported_text)
        self.assertIn("Request: REQ001", exported_text)
        self.assertIn("Contractor: Smith & Sons Construction", exported_text)
        self.assertIn("Total Quote: $1458.00", exported_text)

    def test_8_missing_optional_fields(self):
        res = {
            "decision": "AUTO_QUOTE"
        }
        log_decision("Minimal result dict", res, log_file_path=self.log_path)

        df = pd.read_csv(self.log_path, dtype=str).fillna("")
        self.assertEqual(df.iloc[0]["contractor_name"], "")
        self.assertEqual(df.iloc[0]["equipment_name"], "")
        self.assertEqual(df.iloc[0]["score"], "")
        self.assertEqual(df.iloc[0]["total_quote"], "")

    def test_9_missing_log_file_generation(self):
        new_log = Path(self.temp_dir.name) / "subfolder" / "new_log.csv"
        res = {"decision": "REQUEST_INFO"}
        log_decision("Prompt", res, log_file_path=new_log)

        self.assertTrue(new_log.exists())
        df = pd.read_csv(new_log, dtype=str)
        self.assertEqual(len(df), 1)

    def test_10_malformed_incomplete_result_dictionary(self):
        log_decision("Malformed", None, log_file_path=self.log_path)  # type: ignore
        log_decision("Malformed string", "not_a_dict", log_file_path=self.log_path)  # type: ignore

        df = pd.read_csv(self.log_path, dtype=str).fillna("")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["decision"], "UNRECOGNIZED")


if __name__ == "__main__":
    unittest.main()
