"""Guard independent eligibility, two sampling budgets, and partial-batch resumption."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.h2_data import digest
from src.h2_followup_metrics import crossfit, describe
from src.run_h2_followup import jobs_for, load_saved


class FollowupTests(unittest.TestCase):
    def test_floor_and_ceiling_do_not_get_conditional_scores(self):
        for value in (0, 1):
            s = describe(np.full((6, 3, 32), value), seed=1)
            self.assertIsNone(s["inv_conditional"])
            self.assertIsNone(s["crossfit"]["inv"])

    def test_crossfit_gates_on_opposite_draws(self):
        y = np.zeros((1, 3, 8))
        y[0, 1, 6:] = 1
        y[0, 2, 4:] = 1
        s = crossfit(y)
        # Only the second half qualifies; it selects the all-zero first half for measurement.
        self.assertEqual(s["n_fold_contributions"], 1)
        self.assertEqual(s["inv"], 1)
        self.assertEqual(s["eligibility_agreement"], 0)
        with self.assertRaises(ValueError): crossfit(np.zeros((1, 3, 7)))

    def test_accuracy_difference_respects_problem_pairing(self):
        y = np.ones((8, 3, 32))
        y[:, 1:, 16:] = 0
        s = describe(y, seed=1)
        self.assertEqual(s["other_minus_canonical"], -0.5)
        self.assertEqual(s["accuracy_difference_problem_bootstrap_ci95"], [-0.5, -0.5])

    def fixtures(self):
        p = {"sha256": "frozen", "shots": [], "settings": {"samples": {"strict": 32, "symbolic": 8}, "batch_size": 64}}
        b = {"items": [{"arm": arm, "item_id": f"{arm}/{i}", "question": "How many?", "gold": "2",
                          "audit_expression": "NEVER_INCLUDE_THIS"}
                         for i, arm in enumerate(("strict", "strict", "strict", "symbolic"))]}
        return p, b

    def test_budgets_and_no_answer_program_leak(self):
        p, b = self.fixtures()
        jobs = jobs_for(p, b)
        self.assertEqual(len(jobs), 104)
        self.assertEqual(sum(j["item_id"].startswith("strict") for j in jobs), 96)
        for j in jobs:
            self.assertNotIn("NEVER_INCLUDE_THIS", j["prompt"])
            self.assertNotIn("2", j["prompt"])

    def test_partial_last_batch_resumption(self):
        p, b = self.fixtures()
        jobs = jobs_for(p, b)
        with tempfile.TemporaryDirectory() as tmp, patch("src.run_h2_followup.RUN_DIR", Path(tmp)):
            folder = Path(tmp)/"a__b"; folder.mkdir()
            for index, chunk in enumerate((jobs[:64], jobs[64:])):
                if index == 1:
                    self.assertEqual(load_saved(p, b, "a/b", require_complete=False)[3], [1])
                    with self.assertRaises(ValueError): load_saved(p, b, "a/b")
                records = [{"item_id": j["item_id"], "sample": j["sample"], "gold": "2", "generation": "#### 2",
                            "pred": "2", "correct": True} for j in chunk]
                value = {"model": "a/b", "protocol_sha256": "frozen", "records": records,
                         "records_sha256": digest(records)}
                (folder/f"batch_{index:04d}.json").write_text(json.dumps(value))
            loaded = load_saved(p, b, "a/b")
            self.assertEqual(len(loaded[2]), 104)
            self.assertEqual(loaded[3], [])


if __name__ == "__main__":
    unittest.main()
