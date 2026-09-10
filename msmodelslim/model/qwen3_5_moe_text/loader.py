# -*- coding: UTF-8 -*-

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class Qwen3_5MoeTextAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.qwen3_5_moe_text.model_adapter:Qwen3_5MoeTextModelAdapter"
