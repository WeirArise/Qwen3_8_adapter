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
Tests for the text-only adapter and its registration.

Only the structural pieces are covered here; the calibration forward path needs
the checkpoint and an Ascend device and is explicitly out of scope (see the
module docstring of `model_adapter.py`).
"""

import configparser
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from msmodelslim.model.qwen3_5_moe_text.model_adapter import Qwen3_5MoeTextModelAdapter

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_INI = REPO_ROOT / "config" / "config.ini"


def _bare_adapter() -> Qwen3_5MoeTextModelAdapter:
    """An instance with no config loaded, for methods that do not need one."""
    return object.__new__(Qwen3_5MoeTextModelAdapter)


class TestModelClassSubstitution(unittest.TestCase):
    """The dispatch reaches the multimodal branch; the class is swapped here."""

    def test_conditional_generation_maps_to_causal_lm(self):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        self.assertIs(
            Qwen3_5MoeTextModelAdapter._text_model_cls_for(moe.Qwen3_5MoeForConditionalGeneration),
            moe.Qwen3_5MoeForCausalLM,
        )

    def test_dense_conditional_generation_maps_to_dense_causal_lm(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as base

        self.assertIs(
            Qwen3_5MoeTextModelAdapter._text_model_cls_for(base.Qwen3_5ForConditionalGeneration),
            base.Qwen3_5ForCausalLM,
        )

    def test_an_unrelated_class_is_left_alone(self):
        class _Other:
            pass

        self.assertIsNone(Qwen3_5MoeTextModelAdapter._text_model_cls_for(_Other))

    def test_the_substituted_class_exists_in_transformers(self):
        """Guards against a rename on the transformers side."""
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        target = Qwen3_5MoeTextModelAdapter._text_model_cls_for(moe.Qwen3_5MoeForConditionalGeneration)
        self.assertTrue(hasattr(target, "__init__"))
        self.assertEqual(target.__name__, "Qwen3_5MoeForCausalLM")


class TestTextPositionIds(unittest.TestCase):
    """Convention checks; equivalence to the model is covered in test_position_ids.py."""

    def test_positions_count_from_zero(self):
        adapter = _bare_adapter()
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        mask = torch.ones(1, 5, dtype=torch.long)

        position_ids, rope_deltas = adapter._text_position_ids(input_ids, mask)

        self.assertEqual(position_ids.tolist(), [[0, 1, 2, 3, 4]])
        self.assertIsNone(rope_deltas)

    def test_every_batch_row_gets_its_own_positions(self):
        adapter = _bare_adapter()
        input_ids = torch.zeros(3, 4, dtype=torch.long)

        position_ids, _ = adapter._text_position_ids(input_ids, None)

        self.assertEqual(position_ids.shape, (3, 4))
        self.assertEqual(position_ids.tolist(), [[0, 1, 2, 3]] * 3)

    def test_positions_are_independent_of_the_attention_mask(self):
        """Matches the model's own inference, which ignores the mask."""
        adapter = _bare_adapter()
        input_ids = torch.zeros(1, 5, dtype=torch.long)

        padded, _ = adapter._text_position_ids(input_ids, torch.tensor([[0, 0, 1, 1, 1]]))
        plain, _ = adapter._text_position_ids(input_ids, torch.ones(1, 5, dtype=torch.long))

        self.assertEqual(padded.tolist(), plain.tolist())


class TestVisionIsSkipped(unittest.TestCase):
    def test_visit_does_not_yield_a_vision_request(self):
        adapter = _bare_adapter()

        sentinel = object()

        def _layers(model):
            yield "model.language_model.layers.0", sentinel

        adapter.generate_decoder_layer = _layers
        model = object()

        with patch(
            "msmodelslim.model.qwen3_5_moe_text.model_adapter.generated_decoder_layer_visit_func",
            return_value=iter(()),
        ) as patched:
            list(adapter.generate_model_visit(model))

        patched.assert_called_once()
        called_with = patched.call_args
        self.assertIs(called_with.args[0], model)
        self.assertIn("transformer_blocks", called_with.kwargs)


class TestRegistrationInConfigIni(unittest.TestCase):
    """The model name, the loader path and the dependency must all be present."""

    @classmethod
    def setUpClass(cls):
        parser = configparser.ConfigParser()
        parser.read(CONFIG_INI)
        cls.parser = parser

    def test_model_names_registered(self):
        self.assertIn("qwen3_5_moe_text", self.parser["ModelAdapter"])
        names = self.parser["ModelAdapter"]["qwen3_5_moe_text"]
        self.assertIn("Qwen3.8-2.4T-A95B-FP8", names)
        self.assertIn("Qwen3.8-2.4T-A95B", names)

    def test_multimodal_variant_registered_under_the_existing_adapter(self):
        """Qwen3.8-27B is architecturally identical to the already-supported Qwen3.6-27B."""
        names = self.parser["ModelAdapter"]["qwen3_5_moe"]
        self.assertIn("Qwen3.8-27B", names)

    def test_entry_point_registered_and_importable(self):
        path = self.parser["ModelAdapterEntryPoints"]["qwen3_5_moe_text"]
        module_path, _, class_name = path.partition(":")
        module = importlib.import_module(module_path)
        loader = getattr(module, class_name)
        self.assertEqual(
            loader.ADAPTER_CLASS_PATH,
            "msmodelslim.model.qwen3_5_moe_text.model_adapter:Qwen3_5MoeTextModelAdapter",
        )

    def test_entry_point_target_is_the_class_under_test(self):
        path = self.parser["ModelAdapterEntryPoints"]["qwen3_5_moe_text"]
        module_path, _, class_name = path.partition(":")
        module = importlib.import_module(module_path)
        adapter_path, _, adapter_name = getattr(module, class_name).ADAPTER_CLASS_PATH.partition(":")
        self.assertIs(
            getattr(importlib.import_module(adapter_path), adapter_name),
            Qwen3_5MoeTextModelAdapter,
        )

    def test_dependency_declared(self):
        self.assertIn("qwen3_5_moe_text", self.parser["ModelAdapterDependencies"])

    def test_declared_transformers_version_supplies_the_text_classes(self):
        """The pinned floor must actually expose the classes the adapter needs."""
        import transformers
        from packaging.version import Version

        spec = self.parser["ModelAdapterDependencies"]["qwen3_5_moe_text"]
        floor = Version(spec.split(">=")[1].rstrip("}").strip().strip('"'))
        self.assertGreaterEqual(Version(transformers.__version__), floor)

        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        self.assertTrue(hasattr(moe, "Qwen3_5MoeForCausalLM"))
        self.assertTrue(hasattr(moe, "Qwen3_5MoeTextModel"))


if __name__ == "__main__":
    unittest.main()
