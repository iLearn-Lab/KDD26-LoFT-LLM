"""Check the added configuration layer and SFT supervision without importing models."""
import argparse
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from release_config import ROOT, llm_paths, load_profile, residual_setting, sft_runs, stage_arguments
from train_qlora import encode_example, pad_examples, validate_records


def original_parser(entry):
    parser = argparse.ArgumentParser()
    tree = ast.parse((ROOT / entry).read_bytes())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "parser"
                and node.func.attr == "add_argument"):
            continue
        flags = [ast.literal_eval(arg) for arg in node.args]
        kwargs = {}
        for item in node.keywords:
            kwargs[item.arg] = ({"int": int, "str": str, "bool": bool, "float": float}[item.value.id]
                                if item.arg == "type" else ast.literal_eval(item.value))
        parser.add_argument(*flags, **kwargs)
    return parser


class ConfigurationTests(unittest.TestCase):
    def test_profiles_match_existing_entrypoint_parsers(self):
        entries = {"plfm": "run_plfm.py", "residual": "run_TSLFL.py", "sft": "run_TSLFL_LLM_data.py", "infer": "run_TSLFL.py"}
        for domain in ["fundar", "solar"]:
            for horizon in [1, 3, 10]:
                profile = load_profile(ROOT / "configs" / f"{domain}.json", horizon)
                for stage, entry in entries.items():
                    with self.subTest(domain=domain, horizon=horizon, stage=stage):
                        args = original_parser(entry).parse_args(stage_arguments(profile, stage))
                        self.assertEqual(args.pred_len, horizon)
                        self.assertIn(f"_{horizon}", args.model_id)

    def test_export_checkpoint_matches_original_residual_training_name(self):
        profile = load_profile(ROOT / "configs/fundar.json", 3)
        args = original_parser("run_TSLFL.py").parse_args(stage_arguments(profile, "residual"))
        self.assertEqual(sft_runs(args), [(3, "ProductAmt_30_3", residual_setting(args))])
        self.assertEqual(residual_setting(args),
                         "TSLFL_Forcast_ProductAmt_30_3_iTransformer_TSLFL_ftM_sl30_ll0_pl3_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0")

    def test_domain_adapter_paths_are_distinct(self):
        with patch.dict("os.environ", {}, clear=True):
            finance, solar = llm_paths("fundar"), llm_paths("solar")
        self.assertEqual(finance[0], "Qwen/Qwen3-8B")
        self.assertEqual(solar[0], "Qwen/Qwen3-8B")
        self.assertNotEqual(finance[1], solar[1])

class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) + 10 for c in text]


class SupervisionTests(unittest.TestCase):
    def messages(self, response="[1,2]\n[3,4]"):
        return [{"role": "user", "content": "original domain prompt"}, {"role": "assistant", "content": response}]

    def test_original_export_format_is_supported(self):
        validate_records([{"messages": self.messages()}], "fundar", 2)
        validate_records([{"messages": self.messages("[1,2]")}], "solar", 2)

    def test_wrong_horizon_or_sentinel_format_is_rejected(self):
        for response in ["[1]\n[2]", "[NaN,2]\n[3,4]", "[true,2]\n[3,4]", "answer [1,2]\n[3,4]"]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                validate_records([{"messages": self.messages(response)}], "fundar", 2)

    def test_prompt_is_masked_but_response_and_eos_are_supervised(self):
        example = encode_example(FakeTokenizer(), self.messages(), 200)
        first = next(i for i, value in enumerate(example["labels"]) if value != -100)
        self.assertGreater(first, 0)
        self.assertEqual(example["labels"][first:], example["input_ids"][first:])
        self.assertEqual(example["labels"][-1], 2)

    def test_padding_eos_does_not_remove_actual_eos_label(self):
        a = encode_example(FakeTokenizer(), self.messages("[1]"), 200)
        b = encode_example(FakeTokenizer(), self.messages("[1,2,3,4]"), 200)
        result = pad_examples([a, b], 2)
        self.assertEqual(result["labels"][0][len(a["labels"]) - 1], 2)
        self.assertTrue(all(value == -100 for value in result["labels"][0][len(a["labels"]):]))

    def test_long_prompt_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            encode_example(FakeTokenizer(), self.messages(), 3)


if __name__ == "__main__":
    unittest.main()
