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
Forward-pass verification that the text-only position ids are correct.

The question this settles: a text-only backbone has no `get_rope_index`, so the
adapter computes the position ids itself.  Do those produce the same activations
the model would have produced on its own?

No checkpoint is needed to answer it.  A small model with the same architecture
is built from `Qwen3_5MoeTextConfig` -- the layer pattern, MoE routing, hybrid
attention and rotary setup are all the real code paths, just with small
dimensions -- and run twice: once with `position_ids=None`, letting the model
infer them, and once with the adapter's values.  The activations must be
identical.

This runs on CPU in a couple of seconds, which is the point: the property is
checkable in CI rather than only on the machine that holds a 2.4T checkpoint.
"""

import unittest

import torch

from msmodelslim.model.qwen3_5_moe_text.model_adapter import Qwen3_5MoeTextModelAdapter


def _small_config():
    """A scaled-down but structurally faithful text-only Qwen3.5-MoE config."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

    config = moe.Qwen3_5MoeTextConfig(
        vocab_size=128,
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
        # Same ratio as Qwen3.8: three linear-attention layers per full one.
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
    return config


def _build_model():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

    torch.manual_seed(0)
    model = moe.Qwen3_5MoeTextModel(_small_config()).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _adapter():
    return object.__new__(Qwen3_5MoeTextModelAdapter)


class TestSmallModelIsStructurallyFaithful(unittest.TestCase):
    """Guard the fixture itself: a toy config must still exercise the real paths."""

    def test_layer_pattern_matches_the_full_attention_interval(self):
        config = _small_config()
        interval = config.full_attention_interval
        self.assertEqual(len(config.layer_types), config.num_hidden_layers)
        for idx, kind in enumerate(config.layer_types):
            expected = "full_attention" if (idx + 1) % interval == 0 else "linear_attention"
            self.assertEqual(kind, expected, f"layer {idx}")

    def test_config_has_no_vision_tower(self):
        self.assertFalse(hasattr(_small_config(), "vision_config"))

    def test_model_is_small_enough_for_ci(self):
        self.assertLess(sum(p.numel() for p in _build_model().parameters()), 2_000_000)


class TestPositionIdsMatchAutoInferred(unittest.TestCase):
    """The adapter's position ids must reproduce the model's own inference."""

    @classmethod
    def setUpClass(cls):
        cls.model = _build_model()
        cls.adapter = _adapter()

    def _forward(self, input_ids, attention_mask, use_adapter_positions):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if use_adapter_positions:
            positions, deltas = self.adapter._text_position_ids(input_ids, attention_mask)
            self.assertIsNone(deltas, "a text-only backbone reports no rope delta")
            # The shared forward expands a 2-D result to the three mROPE sections.
            kwargs["position_ids"] = positions[None, ...].expand(3, positions.shape[0], -1)
        with torch.no_grad():
            return self.model(**kwargs).last_hidden_state

    def _assert_identical(self, attention_mask):
        input_ids = torch.randint(0, 128, attention_mask.shape)
        reference = self._forward(input_ids, attention_mask, use_adapter_positions=False)
        actual = self._forward(input_ids, attention_mask, use_adapter_positions=True)
        self.assertTrue(
            torch.equal(reference, actual),
            f"max abs difference {float((reference - actual).abs().max()):.3e}",
        )

    def test_without_padding(self):
        self._assert_identical(torch.ones(2, 8, dtype=torch.long))

    def test_with_right_padding(self):
        self._assert_identical(torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]]))

    def test_with_left_padding(self):
        self._assert_identical(torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]]))

    def test_batch_of_one(self):
        self._assert_identical(torch.ones(1, 5, dtype=torch.long))


class TestPositionIdConvention(unittest.TestCase):
    """Pin the convention, since the obvious alternative is subtly wrong.

    `cumsum(attention_mask) - 1` is the convention for left-padded *generation*.
    It is not what this model infers.  Using it would quietly shift activations
    on padded calibration batches -- measured at ~3e-1 max abs difference on the
    right-padded case -- so the choice is asserted rather than left implicit.
    """

    def setUp(self):
        self.adapter = _adapter()

    def test_positions_are_a_plain_range(self):
        input_ids = torch.zeros(2, 6, dtype=torch.long)
        positions, _ = self.adapter._text_position_ids(input_ids, torch.ones(2, 6, dtype=torch.long))
        self.assertEqual(positions.tolist(), [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    def test_attention_mask_does_not_change_the_positions(self):
        """The model's own inference ignores the mask; so does the adapter's."""
        input_ids = torch.zeros(2, 6, dtype=torch.long)
        padded = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
        with_mask, _ = self.adapter._text_position_ids(input_ids, padded)
        without, _ = self.adapter._text_position_ids(input_ids, None)
        self.assertEqual(with_mask.tolist(), without.tolist())

    def test_convention_differs_from_the_generation_style_cumsum(self):
        """Documents the rejected alternative, so the choice is not silently reverted."""
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        input_ids = torch.zeros(1, 6, dtype=torch.long)

        positions, _ = self.adapter._text_position_ids(input_ids, attention_mask)
        cumsum = attention_mask.long().cumsum(-1) - 1
        cumsum.masked_fill_(attention_mask == 0, 1)

        self.assertNotEqual(positions.tolist(), cumsum.tolist())
        self.assertEqual(cumsum.tolist(), [[0, 1, 2, 1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
