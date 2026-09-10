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
Tests for the model-graph shim.

Half of these tests assert against the *installed* `transformers` classes rather
than fixtures, because the whole premise of the shim is a structural claim about
how `transformers.models.qwen3_5_moe` builds the two model variants.  Class
introspection needs no weights, so an upstream `transformers` bump that changes
the assumption fails here immediately instead of at model-load time on a machine
that has the checkpoint.

The remainder uses small stand-ins modelled on those classes, so the shim's
behaviour is covered without instantiating a 2.4T-parameter model.
"""

import unittest

from torch import nn

from msmodelslim.model.qwen3_5_moe_text.model_shim import (
    BACKBONE_ATTRIBUTE,
    attach_language_model_alias,
    is_text_only_graph,
)


# --------------------------------------------------------------------------- #
# Stand-ins mirroring the real class layout
# --------------------------------------------------------------------------- #
class _TextBackbone:
    """Stands in for the text backbone (Qwen3_5MoeTextModel)."""

    def __init__(self):
        self.embed_tokens = "embed_tokens"
        self.layers = ["layer0", "layer1", "layer2"]
        self.norm = "norm"
        self.rotary_emb = "rotary_emb"
        self.config = "text-config"


class _VlmShell:
    """Stands in for the multimodal shell (Qwen3_5MoeModel)."""

    def __init__(self):
        self.visual = "vision-tower"
        self.language_model = _TextBackbone()


class _CausalLM:
    """Stands in for Qwen3_5MoeForCausalLM: backbone directly under `.model`."""

    def __init__(self):
        self.model = _TextBackbone()
        self.lm_head = "lm_head"


class _ConditionalGeneration:
    """Stands in for Qwen3_5MoeForConditionalGeneration."""

    def __init__(self):
        self.model = _VlmShell()
        self.lm_head = "lm_head"


class TestRealTransformersStructure(unittest.TestCase):
    """Verify the structural premise against the installed transformers."""

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

        cls.moe = moe

    def test_both_variants_place_the_backbone_under_model(self):
        for name in ("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"):
            with self.subTest(model=name):
                self.assertTrue(hasattr(self.moe, name))

    def test_multimodal_shell_wraps_a_text_model_under_language_model(self):
        """The alias is only sound because both sides are the same class."""
        import inspect

        src = inspect.getsource(self.moe.Qwen3_5MoeModel.__init__)
        self.assertIn("self.language_model = Qwen3_5MoeTextModel", src)
        self.assertIn("self.visual = Qwen3_5MoeVisionModel", src)

    def test_causal_lm_uses_the_text_model_directly(self):
        import inspect

        src = inspect.getsource(self.moe.Qwen3_5MoeForCausalLM.__init__)
        self.assertIn("self.model = Qwen3_5MoeTextModel", src)

    def test_text_model_assigns_every_attribute_the_adapter_reaches_for(self):
        """The backbone's members are instance attributes, so read the constructor."""
        import inspect

        src = inspect.getsource(self.moe.Qwen3_5MoeTextModel.__init__)
        for attr in ("layers", "embed_tokens", "norm", "rotary_emb"):
            with self.subTest(attribute=attr):
                self.assertIn(
                    f"self.{attr} =",
                    src,
                    f"Qwen3_5MoeTextModel no longer sets '{attr}'; "
                    f"the alias no longer covers the adapter's access path",
                )

    def test_rope_index_exists_only_on_the_multimodal_shell(self):
        """Documents why position ids need separate handling, not an alias."""
        self.assertTrue(hasattr(self.moe.Qwen3_5MoeModel, "get_rope_index"))
        self.assertFalse(hasattr(self.moe.Qwen3_5MoeTextModel, "get_rope_index"))


class TestGraphDetection(unittest.TestCase):
    def test_text_only_graph_detected(self):
        self.assertTrue(is_text_only_graph(_CausalLM()))

    def test_multimodal_graph_not_detected(self):
        self.assertFalse(is_text_only_graph(_ConditionalGeneration()))

    def test_object_without_inner_model_is_not_detected(self):
        self.assertFalse(is_text_only_graph(object()))


class TestAliasApplied(unittest.TestCase):
    def setUp(self):
        self.model = _CausalLM()
        self.backbone = self.model.model
        self.report = attach_language_model_alias(self.model)

    def test_alias_is_a_self_reference(self):
        self.assertIs(self.model.model.language_model, self.backbone)
        self.assertTrue(self.report.applied)
        self.assertFalse(self.report.already_aliased)

    def test_every_adapter_access_path_resolves(self):
        inner = self.model.model
        for attr in ("layers", "embed_tokens", "norm", "rotary_emb", "config"):
            with self.subTest(attribute=attr):
                self.assertEqual(
                    getattr(inner.language_model, attr),
                    getattr(inner, attr),
                )

    def test_backbone_type_recorded(self):
        self.assertEqual(self.report.backbone_type, "_TextBackbone")


class TestAliasIsSafe(unittest.TestCase):
    def test_multimodal_graph_left_untouched(self):
        model = _ConditionalGeneration()
        original = model.model.language_model
        report = attach_language_model_alias(model)
        self.assertFalse(report.applied)
        self.assertTrue(report.already_aliased)
        self.assertIs(model.model.language_model, original)

    def test_second_call_is_idempotent(self):
        model = _CausalLM()
        first = attach_language_model_alias(model)
        second = attach_language_model_alias(model)
        self.assertTrue(first.applied)
        self.assertFalse(second.applied)
        self.assertTrue(second.already_aliased)
        self.assertIs(model.model.language_model, model.model)

    def test_object_without_backbone_is_ignored(self):
        report = attach_language_model_alias(object())
        self.assertFalse(report.applied)


class TestFailureBeingFixed(unittest.TestCase):
    """Pin the pre-shim AttributeError so the shim cannot become vacuous."""

    def test_access_raises_before_alias(self):
        model = _CausalLM()
        with self.assertRaises(AttributeError):
            model.model.language_model.layers

    def test_access_succeeds_after_alias(self):
        model = _CausalLM()
        attach_language_model_alias(model)
        self.assertEqual(len(model.model.language_model.layers), 3)


class TestBackboneAttributeName(unittest.TestCase):
    def test_constant_matches_the_transformers_attribute(self):
        self.assertEqual(BACKBONE_ATTRIBUTE, "language_model")


class TestTorchModuleSafety(unittest.TestCase):
    """The alias must not register the backbone as a child of itself.

    A real `nn.Module` is used here rather than a stand-in, because the hazard
    lives in `nn.Module.__setattr__`: assigning a module to an attribute of
    itself puts it into its own `_modules`, and every tree walk that lacks a
    visited-set then recurses until the interpreter gives up.
    """

    def _module(self):
        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
                self.embed_tokens = nn.Embedding(4, 2)

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Backbone()

        return _Wrapper()

    def test_alias_is_not_registered_as_a_child_module(self):
        model = self._module()
        attach_language_model_alias(model)
        self.assertNotIn("language_model", model.model._modules)

    def test_module_tree_is_not_duplicated(self):
        model = self._module()
        before = len(list(model.named_modules()))
        attach_language_model_alias(model)
        self.assertEqual(len(list(model.named_modules())), before)

    def test_state_dict_is_unaffected(self):
        model = self._module()
        before = set(model.state_dict().keys())
        attach_language_model_alias(model)
        self.assertEqual(set(model.state_dict().keys()), before)

    def test_tree_walks_do_not_recurse(self):
        model = self._module()
        attach_language_model_alias(model)
        model.eval()          # nn.Module._apply walks children()
        model.to("cpu")
        list(model.parameters())

    def test_get_submodule_still_resolves_through_the_alias(self):
        model = self._module()
        attach_language_model_alias(model)
        self.assertIs(model.get_submodule("model.language_model.layers.0"), model.model.layers[0])


if __name__ == "__main__":
    unittest.main()
