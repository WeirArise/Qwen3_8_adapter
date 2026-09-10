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
Tests for the MoE expert conversion used by the text-only loading path.

`convert_experts_to_mlp` turns the packed expert layout that `transformers`
keeps in memory into one MLP per expert.  Two things are worth pinning:

1. that it is *correct* -- the converted block must compute what the original
   computed, and each expert's gate/up/down slices must land in the right place.
   An error here would be silent: the model would still run.
2. which checkpoint layout it can consume, because that decides which published
   checkpoint the adapter can load at all.

On (2), the two published Qwen3.8-2.4T-A95B variants are not interchangeable:

    Qwen3.8-2.4T-A95B       1,609 tensors   experts stored packed
                                            (mlp.experts.gate_up_proj, mlp.experts.down_proj)
    Qwen3.8-2.4T-A95B-FP8 287,119 tensors   experts stored per expert
                                            (mlp.experts.N.{gate,up,down}_proj) plus 142,848
                                            weight_scale_inv block scales

The adapter reads a checkpoint through `_get_state_dict`, which walks the
module's own parameter names and looks each one up in the weight index, skipping
what it does not find.  Those names are the packed ones, so the BF16 variant
matches and the FP8 one does not: its expert weights are simply never found, and
the subsequent `load_state_dict` fails on missing keys.  `test_expected_expert_
layout_is_packed` pins that expectation.
"""

import unittest

import torch

from msmodelslim.model.qwen3_5_moe.moe_utils import (
    Qwen3_5MoeExpertMLP,
    Qwen3_5MoeSparseMoeBlockWithMLP,
    convert_experts_to_mlp,
)


def _config(num_experts: int = 4):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

    config = moe.Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=num_experts,
        num_experts_per_tok=2,
        full_attention_interval=2,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        mtp_num_hidden_layers=0,
    )
    config._attn_implementation = "eager"
    return config


def _initialised_moe_block(config):
    """A MoE block with real weights.

    `Qwen3_5MoeExperts` allocates with `torch.empty` and relies on the
    surrounding model's `post_init` to fill it.  Building the block standalone
    leaves those tensors uninitialised, which reads back as NaN and would make
    any numerical comparison meaningless -- so build the whole model and take the
    block out of it.
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

    torch.manual_seed(0)
    model = moe.Qwen3_5MoeForCausalLM(config).eval()
    return model.model.layers[1].mlp


class TestExpertLayoutExpectation(unittest.TestCase):
    """Pin which on-disk layout the adapter can read."""

    def test_expected_expert_parameters_are_packed(self):
        block = _initialised_moe_block(_config())
        names = {name for name, _ in block.experts.named_parameters()}
        self.assertEqual(names, {"gate_up_proj", "down_proj"})
        self.assertEqual(
            tuple(block.experts.gate_up_proj.shape),
            (4, 2 * 16, 32),
            "packed experts should be [num_experts, 2*intermediate, hidden]",
        )
        self.assertEqual(tuple(block.experts.down_proj.shape), (4, 32, 16))

    def test_per_expert_names_are_absent_from_the_packed_block(self):
        """The FP8 variant's naming would not be found by _get_state_dict."""
        block = _initialised_moe_block(_config())
        names = {name for name, _ in block.experts.named_parameters()}
        for missing in ("0.gate_proj.weight", "0.up_proj.weight", "0.down_proj.weight"):
            self.assertNotIn(missing, names)


class TestConversionCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _config()
        cls.original = _initialised_moe_block(cls.config)
        cls.converted = convert_experts_to_mlp(cls.original, cls.config)
        torch.manual_seed(1)
        cls.inputs = torch.randn(2, 5, cls.config.hidden_size)

    def test_conversion_returns_the_unpacked_block_type(self):
        self.assertIsInstance(self.converted, Qwen3_5MoeSparseMoeBlockWithMLP)

    def test_each_expert_becomes_an_mlp(self):
        experts = list(self.converted.experts)
        self.assertEqual(len(experts), self.config.num_experts)
        self.assertTrue(all(isinstance(e, Qwen3_5MoeExpertMLP) for e in experts))

    def test_output_matches_the_packed_block(self):
        with torch.no_grad():
            reference = self.original(self.inputs)
            actual = self.converted(self.inputs)
        self.assertTrue(torch.allclose(reference, actual, atol=1e-6))
        self.assertTrue(torch.isfinite(actual).all())

    def test_gate_and_up_are_split_from_the_packed_tensor(self):
        for expert_idx in range(self.config.num_experts):
            with self.subTest(expert=expert_idx):
                gate, up = self.original.experts.gate_up_proj[expert_idx].chunk(2, dim=0)
                self.assertTrue(torch.equal(self.converted.experts[expert_idx].gate_proj.weight, gate))
                self.assertTrue(torch.equal(self.converted.experts[expert_idx].up_proj.weight, up))

    def test_down_proj_is_carried_over_unchanged(self):
        for expert_idx in range(self.config.num_experts):
            with self.subTest(expert=expert_idx):
                self.assertTrue(
                    torch.equal(
                        self.converted.experts[expert_idx].down_proj.weight,
                        self.original.experts.down_proj[expert_idx],
                    )
                )

    def test_router_and_shared_expert_are_carried_over(self):
        self.assertTrue(torch.equal(self.converted.gate.weight, self.original.gate.weight))
        for name in ("gate_proj", "up_proj", "down_proj"):
            with self.subTest(module=f"shared_expert.{name}"):
                self.assertTrue(
                    torch.equal(
                        getattr(self.converted.shared_expert, name).weight,
                        getattr(self.original.shared_expert, name).weight,
                    )
                )
        self.assertTrue(
            torch.equal(self.converted.shared_expert_gate.weight, self.original.shared_expert_gate.weight)
        )

    def test_no_parameter_is_left_uninitialised(self):
        for name, param in self.converted.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(param).all(), f"{name} was not copied")


class TestAdapterTriggerCondition(unittest.TestCase):
    """The adapter converts when the block is MoE and not already converted."""

    def test_packed_block_is_converted_and_conversion_is_not_repeated(self):
        config = _config()
        block = _initialised_moe_block(config)
        self.assertFalse(isinstance(block, Qwen3_5MoeSparseMoeBlockWithMLP))

        converted = convert_experts_to_mlp(block, config)
        self.assertTrue(isinstance(converted, Qwen3_5MoeSparseMoeBlockWithMLP))

        # The adapter's own guard: already-converted blocks are left alone.
        self.assertTrue(
            isinstance(converted, Qwen3_5MoeSparseMoeBlockWithMLP)
            and hasattr(converted, "experts")
        )


if __name__ == "__main__":
    unittest.main()
