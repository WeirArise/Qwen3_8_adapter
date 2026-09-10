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

from __future__ import annotations

"""
Model-graph adaptation for text-only Qwen3.5-family MoE checkpoints.

Companion to :mod:`config_shim`.  The config shim fixes the *hyper-parameter*
layout; this module fixes the *module graph* layout.

Why they differ
---------------
`transformers.models.qwen3_5_moe` builds the two variants differently:

    Qwen3_5MoeForConditionalGeneration
        self.model = Qwen3_5MoeModel(config)          # multimodal shell
        #   .visual          -> Qwen3_5MoeVisionModel
        #   .language_model  -> Qwen3_5MoeTextModel

    Qwen3_5MoeForCausalLM
        self.model = Qwen3_5MoeTextModel(config)      # the text backbone itself

Both put the text backbone in `model.model`, but the multimodal shell inserts an
extra `language_model` hop.  The shared adapter hardcodes that hop in ten places
(`model.model.language_model.layers`, `.embed_tokens`, `.rotary_emb`, `.norm`),
so a text-only checkpoint raises `AttributeError` on every one of them.

The fix is a single self-referential alias.  Crucially the two are the *same
class* -- `Qwen3_5MoeModel.language_model` is a `Qwen3_5MoeTextModel` and
`Qwen3_5MoeForCausalLM.model` is also a `Qwen3_5MoeTextModel` -- so the alias
points at an object of exactly the type the adapter expects, with an identical
attribute surface.  No wrapper, no copy, no attribute forwarding.

What this deliberately does not do
----------------------------------
`Qwen3_5MoeTextModel` has no `get_rope_index` / `compute_3d_position_ids` (they
belong to the multimodal shell, which uses vision-aware mROPE).  A text-only
checkpoint needs ordinary 2-D position ids instead.  That is model *behaviour*,
not graph shape, so it is not something an alias can honestly paper over and it
is not attempted here -- see `position_ids.py`.
"""

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "BACKBONE_ATTRIBUTE",
    "GraphReport",
    "is_text_only_graph",
    "attach_language_model_alias",
]


BACKBONE_ATTRIBUTE = "language_model"


@dataclass
class GraphReport:
    """Audit record for the alias, for logging and for tests to assert on."""

    applied: bool = False
    backbone_type: Optional[str] = None
    already_aliased: bool = False

    @property
    def shell_absent(self) -> bool:
        return self.applied and not self.already_aliased


def is_text_only_graph(model: Any) -> bool:
    """True when `model.model` is the text backbone itself, not a VLM shell.

    Detected structurally: a multimodal shell owns a `visual` tower, and its
    backbone sits one hop down under `language_model`.  Neither holds for the
    text-only graph.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return False
    if hasattr(inner, BACKBONE_ATTRIBUTE):
        return False
    # The multimodal shell always carries a vision tower; the text backbone
    # carries the transformer stack directly.
    return hasattr(inner, "layers")


def attach_language_model_alias(model: Any) -> GraphReport:
    """Expose the text backbone under `model.model.language_model`.

    After this call every ``model.model.language_model.<attr>`` lookup used by
    the shared adapter resolves to ``model.model.<attr>``:

        layers, embed_tokens, norm, rotary_emb, config, ...

    Mutates `model` in place and returns an audit record.  Safe to call on a
    multimodal model (no-op) and safe to call twice.
    """
    report = GraphReport()

    inner = getattr(model, "model", None)
    if inner is None:
        return report

    report.backbone_type = type(inner).__name__

    # Check for the multimodal shell *first*: it has no `layers` of its own, so
    # testing for `layers` before this would misclassify it as "not a graph we
    # understand" and lose the already-aliased signal.
    if hasattr(inner, BACKBONE_ATTRIBUTE) and getattr(inner, BACKBONE_ATTRIBUTE) is not None:
        report.already_aliased = True
        return report

    if not hasattr(inner, "layers"):
        return report

    setattr(inner, BACKBONE_ATTRIBUTE, inner)
    report.applied = True
    return report
