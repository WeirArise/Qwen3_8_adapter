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
Text-only Qwen3.5-family MoE adapter.

Wraps the shared `qwen3_5_moe` adapter rather than duplicating it.  The two
shims do the heavy lifting, so only the differences that are genuinely
text-only remain here:

    config_shim  -- flat hyper-parameters presented as the nested VLM layout
    model_shim   -- text backbone exposed under `model.model.language_model`

WHAT IS VERIFIED
----------------
* the config shim and the model shim, against the published `config.json` of
  Qwen3.8-2.4T-A95B and against the installed `transformers` class shape;
* that the backbone alias does not corrupt `nn.Module`'s module tree;
* the overrides below, which are structural and are asserted by unit tests.

WHAT IS NOT VERIFIED
--------------------
The calibration forward path.  `Qwen3_5MoeTextModel` has no `get_rope_index`
(only the multimodal shell does), so the position ids have to be produced
differently -- see :meth:`_text_position_ids`.  Whether that yields the tensor
shape and semantics the backbone's rotary embedding expects can only be settled
by running the checkpoint, which needs the weights and an Ascend device.  Until
that has been done this adapter must not be used to produce published accuracy
numbers.

Status: structural support only.  See README.md for the verification table.
"""

from pathlib import Path
from typing import Any, Generator

import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.qwen3_5_moe.model_adapter import Qwen3_5ModelAdapter
from msmodelslim.model.qwen3_5_moe_text.config_shim import (
    LayoutReport,
    normalize_text_only_config,
)
from msmodelslim.model.qwen3_5_moe_text.model_shim import (
    GraphReport,
    attach_language_model_alias,
)
from msmodelslim.utils.logging import get_logger, logger_setter

__all__ = ["Qwen3_5MoeTextModelAdapter"]


@logger_setter("msmodelslim.model.qwen3_5_moe_text")
class Qwen3_5MoeTextModelAdapter(Qwen3_5ModelAdapter):
    """Adapter for text-only Qwen3.5-family MoE checkpoints (e.g. Qwen3.8-2.4T-A95B)."""

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False) -> None:
        self._layout_report: LayoutReport = LayoutReport()
        self._graph_report: GraphReport = GraphReport()
        super().__init__(model_type, model_path, trust_remote_code)
        # `super().__init__` loads the config, so the shim has to run after it.
        self._layout_report = normalize_text_only_config(self.config)
        if not self._layout_report.applied:
            get_logger().warning(
                "Checkpoint '%s' already uses the nested layout; the text-only shim was not needed.",
                model_type,
            )

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #
    def _create_model_instance(self, model_cls) -> nn.Module:
        """Load the text-only entry point instead of the multimodal one.

        The shared `init_model` dispatches on `config.architectures[0]`, which the
        config shim has remapped to the conditional-generation name so that the
        dispatch is taken.  The class it then hands us is the multimodal one, so
        the substitution happens here -- the narrowest point available without
        copying `init_model`.
        """
        text_cls = self._text_model_cls_for(model_cls)
        if text_cls is not None:
            get_logger().info(
                "Text-only checkpoint: loading %s instead of %s",
                text_cls.__name__,
                model_cls.__name__,
            )
            model_cls = text_cls
        model = super()._create_model_instance(model_cls)
        self._graph_report = attach_language_model_alias(model)
        return model

    @staticmethod
    def _text_model_cls_for(model_cls):
        """Map a conditional-generation class to its causal-LM counterpart."""
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
        from transformers.models.qwen3_5 import modeling_qwen3_5 as base

        mapping = {
            moe.Qwen3_5MoeForConditionalGeneration: moe.Qwen3_5MoeForCausalLM,
            base.Qwen3_5ForConditionalGeneration: base.Qwen3_5ForCausalLM,
        }
        return mapping.get(model_cls)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """Visit decoder layers only -- there is no vision encoder to process."""
        get_logger().info("Processing text decoder layers (text-only checkpoint, no vision encoder)...")
        yield from generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    # ------------------------------------------------------------------ #
    # Position ids
    # ------------------------------------------------------------------ #
    def _text_position_ids(self, input_ids: torch.Tensor, attention_mask: Any):
        """Ordinary left-to-right position ids for a text-only backbone.

        The shared forward path calls `model.model.get_rope_index(...)`, which
        exists only on the multimodal shell and computes vision-aware mROPE
        indices.  A text-only checkpoint needs the plain sequence positions, and
        reports no rope delta.

        NOTE: unverified against a real checkpoint -- see the module docstring.
        """
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
        return position_ids, None
