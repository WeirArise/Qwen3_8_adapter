#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

"""
Tests for the text-only config-layout shim.

Fixtures are the unmodified `config.json` files published with the corresponding
checkpoints, so the tests assert against the real layouts rather than a
hand-made approximation:

    qwen3_8_2_4t_a95b_text_only.json  Qwen/Qwen3.8-2.4T-A95B-FP8  (flat, text-only)
    qwen3_8_27b_multimodal.json       Qwen/Qwen3.8-27B-FP8       (nested, multimodal)
    qwen3_5_397b_multimodal.json      Qwen/Qwen3.5-397B-A17B     (nested, multimodal)

No model weights and no Ascend hardware are required.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from msmodelslim.model.qwen3_5_moe_text.config_shim import (
    TEXT_ARCH_TO_NESTED,
    TextOnlyVisionConfig,
    is_text_only_layout,
    normalize_text_only_config,
)

FIXTURES = Path(__file__).parent / "fixtures"
TEXT_ONLY = FIXTURES / "qwen3_8_2_4t_a95b_text_only.json"
MULTIMODAL_38 = FIXTURES / "qwen3_8_27b_multimodal.json"
MULTIMODAL_35 = FIXTURES / "qwen3_5_397b_multimodal.json"


def load_config(path: Path) -> SimpleNamespace:
    """Mirror the attribute-access shape of a loaded HF config."""
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return SimpleNamespace(**{k: (SimpleNamespace(**v) if isinstance(v, dict) else v) for k, v in raw.items()})


class TestLayoutDetection(unittest.TestCase):
    def test_text_only_checkpoint_detected_when_text_config_absent(self):
        self.assertTrue(is_text_only_layout(load_config(TEXT_ONLY)))

    def test_multimodal_checkpoints_not_detected_as_text_only(self):
        for path in (MULTIMODAL_38, MULTIMODAL_35):
            with self.subTest(fixture=path.name):
                self.assertFalse(is_text_only_layout(load_config(path)))


class TestContractSatisfied(unittest.TestCase):
    """The three properties the shared adapter depends on."""

    def setUp(self):
        self.config = load_config(TEXT_ONLY)
        self.report = normalize_text_only_config(self.config)

    def test_text_config_resolves_to_the_flat_config_itself(self):
        self.assertIs(self.config.text_config, self.config)
        self.assertEqual(self.config.text_config.num_hidden_layers, self.config.num_hidden_layers)
        self.assertEqual(
            self.config.text_config.full_attention_interval,
            self.config.full_attention_interval,
        )

    def test_vision_config_is_present_with_zero_depth(self):
        self.assertIsInstance(self.config.vision_config, TextOnlyVisionConfig)
        self.assertEqual(self.config.vision_config.depth, 0)

    def test_architecture_remapped_to_the_dispatch_name(self):
        self.assertEqual(self.config.architectures[0], "Qwen3_5MoeForConditionalGeneration")
        self.assertEqual(self.report.remapped_architecture, "Qwen3_5MoeForConditionalGeneration")

    def test_original_architecture_preserved_for_round_trip(self):
        self.assertEqual(self.config._text_only_original_architectures, ["Qwen3_5MoeForCausalLM"])
        self.assertEqual(self.report.original_architectures, ["Qwen3_5MoeForCausalLM"])
        self.assertEqual(self.report.original_model_type, "qwen3_5_moe_text")

    def test_writes_the_adapter_makes_land_on_the_flat_config(self):
        """init_model sets num_hidden_layers=1 then restores it, and forces eager attention."""
        origin = self.config.text_config.num_hidden_layers
        self.config.text_config.num_hidden_layers = 1
        self.assertEqual(self.config.num_hidden_layers, 1)
        self.config.text_config.num_hidden_layers = origin
        self.assertEqual(self.config.num_hidden_layers, origin)

        self.config.text_config._attn_implementation = "eager"
        self.assertEqual(self.config._attn_implementation, "eager")


class TestRealAdapterAccessSites(unittest.TestCase):
    """Replay the exact attribute reads the shared adapter performs."""

    # Derived from msmodelslim/model/qwen3_5_moe/model_adapter.py
    ACCESSORS = (
        ("text_config.num_hidden_layers", lambda c: c.text_config.num_hidden_layers),
        ("text_config.full_attention_interval", lambda c: c.text_config.full_attention_interval),
        ("text_config.layer_types", lambda c: c.text_config.layer_types),
        ("text_config.num_experts", lambda c: c.text_config.num_experts),
        ("text_config.num_experts_per_tok", lambda c: c.text_config.num_experts_per_tok),
        ("text_config.moe_intermediate_size", lambda c: c.text_config.moe_intermediate_size),
        ("text_config.mtp_num_hidden_layers", lambda c: c.text_config.mtp_num_hidden_layers),
        ("text_config.num_attention_heads", lambda c: c.text_config.num_attention_heads),
        ("text_config.num_key_value_heads", lambda c: c.text_config.num_key_value_heads),
        ("vision_config.depth", lambda c: c.vision_config.depth),
        ("config.use_cache", lambda c: c.use_cache),
        ("architectures[0]", lambda c: c.architectures[0]),
    )

    def test_every_accessor_resolves_after_normalization(self):
        config = load_config(TEXT_ONLY)
        normalize_text_only_config(config)
        for label, fn in self.ACCESSORS:
            with self.subTest(accessor=label):
                fn(config)  # must not raise

    def test_accessors_raise_before_normalization(self):
        """Pin the failure being fixed, so the shim cannot become vacuous."""
        config = load_config(TEXT_ONLY)
        with self.assertRaises(AttributeError):
            config.text_config.num_hidden_layers
        with self.assertRaises(AttributeError):
            config.vision_config.depth


class TestStructuralFidelity(unittest.TestCase):
    def test_normalization_does_not_alter_architecture_values(self):
        """Only the three documented properties may change; nothing else moves."""
        config = load_config(TEXT_ONLY)
        snapshot = {
            k: v for k, v in vars(config).items() if not isinstance(v, (dict, list, SimpleNamespace))
        }
        added = set(vars(config)) 
        normalize_text_only_config(config)
        for key, value in snapshot.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(config, key), value)
        newly_added = set(vars(config)) - added
        self.assertEqual(
            newly_added,
            {"text_config", "vision_config", "_text_only_original_architectures"},
        )

    def test_layer_types_pattern_matches_full_attention_interval(self):
        """3 linear_attention : 1 full_attention, as the adapter's mapping assumes."""
        config = load_config(TEXT_ONLY)
        interval = config.full_attention_interval
        types = config.layer_types
        self.assertEqual(len(types), config.num_hidden_layers)
        for idx, kind in enumerate(types):
            expected = "full_attention" if (idx + 1) % interval == 0 else "linear_attention"
            self.assertEqual(kind, expected, f"layer {idx}")

    def test_mtp_declared_with_one_layer(self):
        self.assertEqual(load_config(TEXT_ONLY).mtp_num_hidden_layers, 1)


class TestIdempotency(unittest.TestCase):
    def test_multimodal_config_is_left_untouched(self):
        config = load_config(MULTIMODAL_35)
        report = normalize_text_only_config(config)
        self.assertFalse(report.applied)
        self.assertIsNot(config.text_config, config)
        self.assertEqual(config.text_config.model_type, "qwen3_5_moe_text")
        self.assertEqual(config.architectures[0], "Qwen3_5MoeForConditionalGeneration")

    def test_second_call_is_a_noop(self):
        config = load_config(TEXT_ONLY)
        normalize_text_only_config(config)
        before = list(config.architectures)
        report = normalize_text_only_config(config)
        self.assertFalse(report.applied)
        self.assertEqual(config.architectures, before)

    def test_fixture_files_are_not_modified_on_disk(self):
        """The shim is an in-memory view; the checkpoint keeps its own layout."""
        digest_before = TEXT_ONLY.read_bytes()
        config = load_config(TEXT_ONLY)
        normalize_text_only_config(config)
        self.assertEqual(TEXT_ONLY.read_bytes(), digest_before)
        self.assertEqual(json.loads(digest_before)["model_type"], "qwen3_5_moe_text")


class TestArchitectureMap(unittest.TestCase):
    def test_map_covers_both_moe_and_dense_causal_lm_names(self):
        self.assertEqual(TEXT_ARCH_TO_NESTED["Qwen3_5MoeForCausalLM"], "Qwen3_5MoeForConditionalGeneration")
        self.assertEqual(TEXT_ARCH_TO_NESTED["Qwen3_5ForCausalLM"], "Qwen3_5ForConditionalGeneration")


if __name__ == "__main__":
    unittest.main()
