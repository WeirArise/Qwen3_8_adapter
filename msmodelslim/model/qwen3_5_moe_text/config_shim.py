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
Config-layout adaptation for text-only Qwen3.5-family MoE checkpoints.

Background
----------
The `qwen3_5_moe` adapter was written for multimodal checkpoints
(`Qwen3_5MoeForConditionalGeneration`), whose `config.json` nests the language
model hyper-parameters under `text_config` and adds a `vision_config`.

Qwen3.8-2.4T-A95B ships as a *text-only* checkpoint
(`Qwen3_5MoeForCausalLM`, `model_type: qwen3_5_moe_text`).  Its `config.json` is
**flat**: the very same hyper-parameters (hidden_size, num_hidden_layers,
full_attention_interval, layer_types, num_experts, ...) sit at the top level and
neither `text_config` nor `vision_config` exists.

Because the shared adapter reaches for `self.config.text_config.*` in 22 places
and `self.config.vision_config.depth` in two, loading such a checkpoint fails
immediately with `AttributeError` — before any weight is read.

This module makes the flat layout satisfy that contract **without touching the
upstream adapter and without writing anything back to disk**.  It is a pure
in-memory view: the on-disk `config.json` is never modified, so an exported
checkpoint keeps the layout its own runtime expects.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "TEXT_ARCH_TO_NESTED",
    "LayoutReport",
    "is_text_only_layout",
    "normalize_text_only_config",
    "TextOnlyVisionConfig",
]


# A text-only checkpoint names its architecture after the bare causal LM.  The
# shared adapter switches on the *conditional generation* (multimodal) names, so
# the architecture string is remapped to reach the same code path.
TEXT_ARCH_TO_NESTED: Dict[str, str] = {
    "Qwen3_5MoeForCausalLM": "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForCausalLM": "Qwen3_5ForConditionalGeneration",
}


class TextOnlyVisionConfig:
    """Stand-in for the absent `vision_config`.

    Only `depth` is read by the shared adapter (for a log line and for the
    meta-model layer count).  A text-only checkpoint has no vision tower, so
    zero is the truthful value rather than a placeholder.
    """

    __slots__ = ("depth",)

    def __init__(self, depth: int = 0) -> None:
        self.depth = depth


@dataclass
class LayoutReport:
    """What the normalization changed, for logging and for tests to assert on."""

    applied: bool = False
    original_architectures: List[str] = field(default_factory=list)
    original_model_type: Optional[str] = None
    added_text_config: bool = False
    added_vision_config: bool = False
    remapped_architecture: Optional[str] = None

    @property
    def is_text_only(self) -> bool:
        return self.original_model_type is not None and self.original_model_type.endswith("_text")


def is_text_only_layout(config: Any) -> bool:
    """True when `config` uses the flat, text-only layout.

    The discriminator is the absence of `text_config`, not the `model_type`
    suffix: a checkpoint may legitimately be text-only under either name, and
    keying off the structure keeps this working if the model_type is renamed.
    """
    return not hasattr(config, "text_config")


def normalize_text_only_config(config: Any) -> LayoutReport:
    """Make a flat text-only config satisfy the nested multimodal contract.

    Mutates `config` in place (in memory only) and returns an audit record.
    Calling this on an already-nested config is a no-op, so it is safe to invoke
    unconditionally from an adapter constructor.

    Three things are arranged:

    1. ``config.text_config`` becomes a self-reference.  Every
       ``self.config.text_config.<attr>`` lookup therefore resolves to the same
       value as ``self.config.<attr>``, including the two assignments the shared
       adapter performs (``num_hidden_layers = 1`` before loading, then restored,
       and ``_attn_implementation = 'eager'``).
    2. ``config.vision_config`` becomes a :class:`TextOnlyVisionConfig` with
       ``depth = 0``.
    3. ``config.architectures[0]`` is remapped to the conditional-generation
       name so the shared architecture dispatch is taken.

    The originals are preserved on the returned report and, for the
    architecture list, on ``config._text_only_original_architectures`` so the
    checkpoint can be written back with its own layout.
    """
    report = LayoutReport()
    if not is_text_only_layout(config):
        return report

    report.applied = True
    report.original_model_type = getattr(config, "model_type", None)

    architectures = getattr(config, "architectures", None)
    if architectures:
        report.original_architectures = list(architectures)
        # Keep the original around so save paths can restore it verbatim.
        config._text_only_original_architectures = list(architectures)
        remapped = TEXT_ARCH_TO_NESTED.get(architectures[0])
        if remapped is not None:
            architectures[0] = remapped
            report.remapped_architecture = remapped

    # Self-reference: the flat config *is* the text config.
    config.text_config = config
    report.added_text_config = True

    if not hasattr(config, "vision_config") or getattr(config, "vision_config", None) is None:
        config.vision_config = TextOnlyVisionConfig(depth=0)
        report.added_vision_config = True

    return report
