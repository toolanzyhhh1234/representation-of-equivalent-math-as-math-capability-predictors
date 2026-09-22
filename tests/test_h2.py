"""CPU regression guards for H2 scoring, grouping, and resumable batch validation."""
import copy
from fractions import Fraction
import json
import unittest

import numpy as np
import torch
from transformers import BatchEncoding, GenerationConfig, LlamaConfig, LlamaForCausalLM

from src.h2_data import DEST, arithmetic, digest, validate_bundle
from src.h2_metrics import per_problem, summarize
from src.run_h2 import generate_tokens, score, validate_batch


class H2Tests(unittest.TestCase):
    def test_live_generation_api_without_model_download(self):
        class TinyTokenizer:
            def __call__(self, prompts, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[1, 3, 4]] * len(prompts)),
                                      "attention_mask": torch.ones(len(prompts), 3, dtype=torch.long)})
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=32)).eval()
        cfg = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.95, top_k=0,
                               max_new_tokens=2, pad_token_id=0, eos_token_id=2)
        with torch.no_grad(): tokens = generate_tokens(model, TinyTokenizer(), ["toy"], cfg)
        self.assertEqual(tokens.shape[0], 1)
        self.assertIn(tokens.shape[1], (1, 2))

    def test_floor_is_not_conditional_invariance(self):
        s = summarize(np.zeros((4, 3, 8)))
        self.assertEqual(s["inv_raw"], 1)
        self.assertEqual(s["n_always_wrong"], 4)
        self.assertEqual(s["n_eligible"], 0)
        self.assertIsNone(s["inv_conditional"])
        self.assertIsNone(s["conditional_ci95"])

    def test_ceiling_and_empty_inputs(self):
        self.assertEqual(summarize(np.ones((4, 3, 8)))["n_eligible"], 0)
        with self.assertRaises(ValueError): summarize(np.zeros((0, 3, 8)))
        with self.assertRaises(ValueError): summarize(np.zeros((3, 3, 1)))
        with self.assertRaises(ValueError): summarize(np.full((3, 3, 8), np.nan))

    def test_equally_successful_variants(self):
        y = np.tile([0, 0, 0, 0, 1, 1, 1, 1], (6, 3, 1))
        s = summarize(y)
        self.assertEqual(s["accuracy"], 0.5)
        self.assertEqual(s["n_eligible"], 6)
        self.assertEqual(s["inv_conditional"], 1)
        self.assertTrue(s["adequate_conditional_coverage"])

    def test_variant_spread_and_noise_correction(self):
        y = np.array([[[0]*8, [0]*4+[1]*4, [1]*8]])
        s = per_problem(y)
        variance = 1/6
        noise = (2/3)*(0.25/7)/3
        self.assertAlmostEqual(s["sd"][0], np.sqrt(variance))
        self.assertAlmostEqual(s["sd_noise_adjusted"][0], np.sqrt(variance-noise))
        self.assertFalse(summarize(y)["adequate_conditional_coverage"])

    def test_exact_arithmetic_and_reject_code(self):
        self.assertEqual(arithmetic("16*0.25+(36-16)*0.2"), Fraction(8))
        self.assertEqual(arithmetic("1/10+2/10"), Fraction(3, 10))
        with self.assertRaises(ValueError): arithmetic("__import__('os')")
        with self.assertRaises(ValueError): arithmetic("True")

    def test_bundle_integrity_and_gold(self):
        bundle = json.loads(DEST.read_text())
        validate_bundle(bundle)
        changed = copy.deepcopy(bundle)
        changed["items"][0]["question"] += " changed"
        with self.assertRaises(ValueError): validate_bundle(changed)
        changed = copy.deepcopy(bundle)
        changed["items"][0]["gold"] = "999"
        changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaises(ValueError): validate_bundle(changed)

    def test_scoring_uses_existing_harness(self):
        self.assertEqual(score("Work. #### $1,000\nQuestion: 999", "1000"), ("1000", True))
        self.assertEqual(score("No numeric answer", "1"), (None, False))

    def test_resume_rejects_wrong_samples_and_stale_scores(self):
        jobs = [{"item_id": "strict/0/canonical", "sample": 0, "gold": "10"}]
        records = [{**jobs[0], "generation": "#### 10", "pred": "10", "correct": True}]
        batch = {"model": "test", "protocol_sha256": "frozen", "records": records,
                 "records_sha256": digest(records)}
        self.assertEqual(validate_batch(batch, jobs, "frozen", "test"), records)
        with self.assertRaises(ValueError): validate_batch(batch, jobs, "other", "test")
        bad = copy.deepcopy(batch)
        bad["records"][0]["sample"] = 1
        bad["records_sha256"] = digest(bad["records"])
        with self.assertRaises(ValueError): validate_batch(bad, jobs, "frozen", "test")
        bad = copy.deepcopy(batch)
        bad["records"][0]["correct"] = False
        bad["records_sha256"] = digest(bad["records"])
        with self.assertRaises(ValueError): validate_batch(bad, jobs, "frozen", "test")


if __name__ == "__main__":
    unittest.main()
