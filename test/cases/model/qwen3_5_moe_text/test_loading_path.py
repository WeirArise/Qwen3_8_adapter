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
Integration test for the text-only loading path.

Builds a small checkpoint that is laid out like the published
Qwen3.8-2.4T-A95B one -- same file set, same weight-key naming, same flat config
-- and drives the adapter through instantiation, model creation and layer-wise
loading.  Needs no Ascend device and no real weights; the small model runs on
CPU.

The naming detail this pins down is easy to get wrong.  The published
checkpoint's 287,119-entry index contains **no** `model.language_model.*` key:
its text backbone is addressed as `model.layers.N`.  The shared adapter builds
`model.language_model.layers.N` and hands it to `get_submodule`, so on a real
checkpoint that lookup only resolves because of the backbone alias.  A fixture
saved straight from `transformers` would hide the problem, because
`save_pretrained` rewrites the keys to the multimodal spelling.
"""

import json
import unittest
from pathlib import Path

import torch

from msmodelslim.model.qwen3_5_moe_text.model_adapter import Qwen3_5MoeTextModelAdapter

# The published checkpoint addresses the backbone without the `language_model`
# hop; a fixture that keeps transformers' multimodal spelling would not exercise
# the alias.
MULTIMODAL_PREFIX = "model.language_model."


def _build_checkpoint(directory: Path) -> Path:
    """Write a small text-only checkpoint in the published layout."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

    directory.mkdir(parents=True, exist_ok=True)

    config = moe.Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        full_attention_interval=4,
        max_position_embeddings=512,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        mtp_num_hidden_layers=0,
    )
    config._attn_implementation = "eager"

    torch.manual_seed(0)
    model = moe.Qwen3_5MoeForCausalLM(config).eval()
    model.save_pretrained(directory, safe_serialization=True)

    # Reshape to the published spelling: no `language_model` hop.
    weights = load_file(str(directory / "model.safetensors"))
    save_file(
        {k.replace(MULTIMODAL_PREFIX, "model.", 1): v for k, v in weights.items()},
        str(directory / "model.safetensors"),
    )
    # `save_pretrained` only writes an index when it shards.  The published
    # checkpoint has 213 shards and therefore always has one, and the adapter
    # reads the index to build its weight map, so synthesise it here.
    index_path = directory / "model.safetensors.index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.is_file()
        else {"metadata": {}, "weight_map": {k: "model.safetensors" for k in weights}}
    )
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = int(sum(v.numel() * v.element_size() for v in weights.values()))
    index["weight_map"] = {
        k.replace(MULTIMODAL_PREFIX, "model.", 1): v for k, v in index["weight_map"].items()
    }
    index_path.write_text(json.dumps(index))

    # A text-only checkpoint has a tokenizer but no processor config.
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel(vocab={f"tok{i}": i for i in range(256)}, unk_token="tok0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json"), unk_token="tok0", pad_token="tok0", eos_token="tok1"
    ).save_pretrained(directory)

    with safe_open(str(directory / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith(MULTIMODAL_PREFIX):
                raise AssertionError(f"fixture still uses the multimodal spelling: {key}")
    return directory


class TestCheckpointFixture(unittest.TestCase):
    """The fixture must differ from a plain transformers save, or it proves nothing."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = _build_checkpoint(Path(cls._tmp.name) / "ckpt")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_weight_keys_use_the_published_naming(self):
        from safetensors import safe_open

        with safe_open(str(self.ckpt / "model.safetensors"), framework="pt") as handle:
            keys = list(handle.keys())
        self.assertTrue(any(k.startswith("model.layers.") for k in keys))
        self.assertFalse(any("language_model" in k for k in keys))

    def test_config_is_flat_and_text_only(self):
        raw = json.loads((self.ckpt / "config.json").read_text())
        self.assertEqual(raw["model_type"], "qwen3_5_moe_text")
        self.assertNotIn("text_config", raw)
        self.assertNotIn("vision_config", raw)

    def test_checkpoint_has_a_tokenizer_and_no_processor(self):
        names = {p.name for p in self.ckpt.iterdir()}
        self.assertIn("tokenizer.json", names)
        self.assertNotIn("preprocessor_config.json", names)


class TestAdapterLoadingPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = _build_checkpoint(Path(cls._tmp.name) / "ckpt")
        cls.adapter = Qwen3_5MoeTextModelAdapter("qwen3_5_moe_text", cls.ckpt, trust_remote_code=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_adapter_constructs_and_shims_the_config(self):
        self.assertTrue(self.adapter.config.text_config is self.adapter.config)
        self.assertEqual(self.adapter.config.vision_config.depth, 0)
        self.assertEqual(self.adapter.config.architectures[0], "Qwen3_5MoeForConditionalGeneration")

    def test_tokenizer_is_recovered_from_a_processorless_checkpoint(self):
        self.assertIsNotNone(self.adapter.tokenizer)
        self.assertTrue(hasattr(self.adapter.tokenizer, "encode"))

    def test_weight_map_is_read_from_the_index(self):
        weight_map = self.adapter._get_weight_map()
        self.assertTrue(weight_map)
        self.assertTrue(all(k in weight_map for k in weight_map))

    def test_model_is_built_from_the_text_only_entry_point(self):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        model = self.adapter._create_model_instance(moe.Qwen3_5MoeForConditionalGeneration)
        self.assertIsInstance(model, moe.Qwen3_5MoeForCausalLM)
        self.assertIsInstance(model.model, moe.Qwen3_5MoeTextModel)

    def test_backbone_alias_resolves_the_adapter_layer_path(self):
        """The lookup the shared adapter performs must reach the real module."""
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        model = self.adapter._create_model_instance(moe.Qwen3_5MoeForConditionalGeneration)
        for idx in (0, 3):
            with self.subTest(layer=idx):
                via_alias = model.get_submodule(f"model.language_model.layers.{idx}")
                real = model.get_submodule(f"model.layers.{idx}")
                self.assertIs(via_alias, real)

    def test_layer_wise_loading_loads_a_decoder_layer(self):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        model = self.adapter._create_model_instance(moe.Qwen3_5MoeForConditionalGeneration)
        layer = self.adapter._load_decoder_if_not_exist(model, "model.language_model.layers.1", 1)
        self.assertIsNotNone(layer)
        self.assertTrue(hasattr(layer.mlp, "experts"), "a MoE layer should expose experts")


if __name__ == "__main__":
    unittest.main()
