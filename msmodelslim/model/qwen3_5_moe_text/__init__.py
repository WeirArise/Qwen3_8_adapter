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
Text-only Qwen3.5-family MoE support.

Qwen3.8-2.4T-A95B ships as a text-only checkpoint (`Qwen3_5MoeForCausalLM`,
`model_type: qwen3_5_moe_text`) whose `config.json` is flat, while the shared
`qwen3_5_moe` adapter expects the multimodal layout (`text_config` nesting plus a
`vision_config`).  This package adapts the layout so the shared architecture
handling can be reused unchanged.
"""

from msmodelslim.model.qwen3_5_moe_text.config_shim import (
    LayoutReport,
    TextOnlyVisionConfig,
    is_text_only_layout,
    normalize_text_only_config,
)
from msmodelslim.model.qwen3_5_moe_text.model_shim import (
    GraphReport,
    attach_language_model_alias,
    is_text_only_graph,
)

__all__ = [
    "LayoutReport",
    "TextOnlyVisionConfig",
    "is_text_only_layout",
    "normalize_text_only_config",
    "GraphReport",
    "attach_language_model_alias",
    "is_text_only_graph",
]
