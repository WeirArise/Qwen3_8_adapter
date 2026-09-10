# Qwen3.8 支持

> **本仓库是 [Ascend/msmodelslim](https://gitcode.com/Ascend/msmodelslim) 的二次开发分支。**
> 上游代码版权归 **Huawei Technologies Co.,Ltd.** 所有，遵循[木兰宽松许可证 v2](./LICENSE)。
> 本分支的全部提交都在上游 `master` 之上，上游历史未作任何改写。

## 1. 为什么需要这个分支

上游的[模型支持矩阵](./docs/zh/knowledge_base/model/README.md)覆盖 Qwen3 / Qwen3.5 / Qwen3.6，但**没有 Qwen3.8**。而 Qwen3.8 的两种形态各自撞上不同的问题：

| 模型 | 情况 |
|---|---|
| **Qwen3.8-27B**（Dense VLM） | 与上游**已支持**的 Qwen3.6-27B 架构完全相同 → **无需新代码**，只需注册 |
| **Qwen3.8-2.4T-A95B**（纯文本 MoE） | 上游**直接崩溃** → 需要真正的适配 |

## 2. Qwen3.8-27B：配置逐键比对

把 `Qwen3.8-27B-FP8` 与 `Qwen3.6-27B` 的官方 `config.json` 展平后逐键比对：

| | 结果 |
|---|---|
| 相同键 | **60** |
| 不同键 | 6 |
| 差异来源 | 全部来自 `quantization_config`（FP8 vs BF16）与 `transformers_version` |

**结论**：两者架构相同。`Qwen3.8-27B` 已加入 `config.ini` 的 `qwen3_5_moe` 注册行，复用上游适配器。

## 3. Qwen3.8-2.4T-A95B：三处不兼容

### 3.1 问题

上游 `qwen3_5_moe` 适配器面向**多模态** checkpoint 编写，而 Qwen3.8-2.4T 是**纯文本**的：

| 项 | 上游假设（多模态） | Qwen3.8-2.4T 实际（纯文本） |
|---|---|---|
| `architectures[0]` | `Qwen3_5MoeForConditionalGeneration` | **`Qwen3_5MoeForCausalLM`** |
| `model_type` | `qwen3_5_moe` | **`qwen3_5_moe_text`**（上游全仓零命中） |
| config 布局 | 嵌套（`text_config` + `vision_config`） | **扁平**（42 个顶层键，两者都没有） |
| 主干路径 | `model.model.language_model.layers.*` | **`model.model.layers.*`** |
| 位置编码 | `model.model.get_rope_index`（视觉感知 mROPE） | **该方法不存在** |

失败是可复现的：上游适配器访问 `self.config.text_config.*` **22 次**、`config.vision_config.depth` **2 次**，且无回退分支；`architectures` 也不在上游接受的两个名字里。

### 3.2 解法：适配契约，而不是复制代码

三处问题分成两层，各由一个小模块解决：

```
config_shim.py   ~160 行   参数布局：扁平 -> 嵌套契约
model_shim.py    ~130 行   模块图：补上缺失的 language_model 一跳
model_adapter.py ~110 行   仅纯文本真实差异的覆盖
```

**为什么 shim 成立，而不是权宜之计**

- `config_shim`：`text_config` 设为**自引用**。上游的 22 处读、2 处**写**（临时把层数设为 1、强制 eager 注意力）全部落到扁平配置上。`vision_config` 用 `depth = 0` 的替身 —— 纯文本模型没有视觉塔，**0 是真值**，不是占位符。
- `model_shim`：别名成立是因为 `transformers` 里**两边是同一个类**：
  ```python
  Qwen3_5MoeModel.__init__        self.language_model = Qwen3_5MoeTextModel._from_config(...)
  Qwen3_5MoeForCausalLM.__init__  self.model          = Qwen3_5MoeTextModel(config)
  ```
  所以别名指向的对象类型完全正确，不需要包装器或属性转发。

**与"复制一份适配器"的对比**

| | 复制适配器 | 本分支 |
|---|---|---|
| 代码量 | ~1000 行 | **~400 行** |
| 依赖什么 | 上游**全部代码** | ① 模型 config 契约 ② transformers 类结构 |
| 上游改 `model_adapter.py` 时 | 每次手工移植 | **零影响** |

上游 `model_adapter.py` 近 7 个月改了 14 次（最近一次是发布前一天，改量 +57/−4），还发生过一次 +170/−205 的重构。本分支的两层 shim **一行都不碰该文件**。

## 4. 验证状态

### 4.1 已验证（可复现，**不需要昇腾硬件**）

| 项 | 结论 | 依据 |
|---|---|---|
| 配置布局适配 | 通过 | 用**官方发布**的 3 份 `config.json` 作 fixture；重放适配器真实的 12 个属性访问点 |
| 模块图适配 | 通过 | 断言**已安装的** `transformers` 类结构，上游改结构会立刻失败 |
| `nn.Module` 安全性 | 通过 | 用真实 `nn.Module` 断言别名未进入 `_modules`、`state_dict()`/`eval()` 不递归 |
| 注册一致性 | 通过 | 断言 `config.ini` 的 loader 路径可导入、指向本适配器、版本下界真的提供所需类 |
| 上游回归 | **与纯净基线逐字相同** | `test/cases/format` + `test/cases/core`：2 failed / 1404 passed / 43 skipped / 1 error（3 个失败为上游预先存在的环境性问题） |
| **位置编码正确性** | **通过（真实前向）** | 用小尺寸同构模型跑真实前向，对比「模型自动推断位置」与「适配器计算的位置」—— **逐元素完全相同**，覆盖无 padding / 右 padding / 左 padding / batch=1 四种情况 |
| **逐层加载路径** | **通过（小尺寸真实 checkpoint）** | 构造符合官方布局的 checkpoint（相同文件集、相同权重键名、相同扁平 config），跑通 `_create_model_instance` → 骨干别名解析 → `_load_decoder_if_not_exist` |
| **MoE 专家融合转换** | **通过（真实前向）** | `convert_experts_to_mlp` 的输出与原打包 MoE 块**逐元素等价**（最大差 ~9e-11）；gate/up 拆分与 down_proj 携带均**逐元素核对**；无参数遗漏 |

```
pytest test/cases/model/qwen3_5_moe_text/          -> 94 passed, 127 subtests
pytest test/cases/format test/cases/core           -> 与官方基线逐字相同
```

### 验证过程中拿到的两个关键事实

**1. 真实 checkpoint 的权重键名（来自官方 287,119 条索引）**

| 命名形态 | 数量 |
|---|---|
| 含 `model.language_model` | **0** |
| 以 `model.layers.` 开头 | **284,027** |

真实 checkpoint 用**纯文本命名** `model.embed_tokens` / `model.layers.N`，而上游适配器构造的是
`model.language_model.layers.N`。**所以 `model_shim` 不是“理论上成立”，而是让逐层加载能工作的
必要条件** —— 实测确认 `get_submodule("model.language_model.layers.0")` 与
`get_submodule("model.layers.0")` 返回**同一个对象**。

> 注意：直接用 `save_pretrained` 保存的 fixture **不能**暴露这个问题 —— `transformers` 会把键名
> 重写成多模态拼写。因此测试里的 fixture 显式改回官方命名，并断言不含 `language_model`。

**2. 纯文本 checkpoint 下 `tokenizer` 会变成 None**

共享构造函数是 `getattr(self._processor, "tokenizer", None)`。多模态 checkpoint 下
`AutoProcessor` 返回一个拥有 tokenizer 的包装器；而纯文本 checkpoint **没有 processor 配置**，
`AutoProcessor` 直接返回 tokenizer 本身，它没有 `.tokenizer` 属性 → `self._tokenizer` 为 None，
而 `adapter.tokenizer` 属性被推理侧读取。已修复：无包装器时直接采用 processor 作为 tokenizer。

**位置编码为何能验证**：纯文本主干没有 `get_rope_index`，但验证它的正确性**不需要权重** —— 用
`Qwen3_5MoeTextConfig` 构造一个小尺寸但结构相同的模型（相同的层模式、MoE 路由、混合注意力、
旋转位置设置），分别用「`position_ids=None`（模型自己推断）」和「适配器的值」跑两次，
激活值必须一致。整个过程 CPU 上几秒，**因此这个性质进入 CI，而不是只能靠拥有 2.4T checkpoint 的机器**。

实验还纠正了一个错选：`cumsum(attention_mask) - 1` 是左 padding **生成**场景的惯例，
而本模型的自动推断用的是 `cache_position`（即 `arange`）。用错会在带 padding 的校准批次上
静默偏移激活值（右 padding 情况下实测最大差 ~3e-1），因此这个选择被测试**钉住**。

### 4.2 未验证 ⚠️

**校准前向路径的其余部分没有验证过。** 具体包括：

- 量化处理器本身（`linear_quant`、`flex_awq_ssz` 等）
- 两个实践配置的量化范围虽从官方排除清单**反推**，但**未在昇腾硬件上跑过**

这些都需要完整规模权重与昇腾设备，因此：

1. 两个实践配置**故意不声明** `verified_model_types` / `verified_tags` —— 那两个字段的含义是
   "已通过项目验证"，而这里没有。
2. **在 4.3 的实测完成之前，不要用本分支产出对外发布的精度/性能数字。**
3. 本分支不提供任何未实测的数字。

### 4.3 实测计划

| # | 指标 | 方法 | 状态 |
|---|---|---|---|
| 1 | ~~位置编码正确性~~ | ~~对比自动推断与适配器计算的位置~~ | ✅ **已完成** |
| 2 | 量化前后 PPL / 下游精度 | 同一模型分别加载 FP8 与量化权重，跑 BoolQ / C-Eval / GSM8K，记录 ΔAcc | `TODO` |
| 3 | 端到端吞吐 | 固定 batch / 序列长度 / 并发，对比 FP8 baseline | `TODO` |
| 4 | 显存占用 | 权重显存与 KV cache 峰值 | `TODO` |
| 5 | ~~逐层加载路径~~ | ~~小尺寸模型上跑通 `init_model` / `generate_decoder_layer`~~ | ✅ **已完成** |
| 6 | ~~MoE 专家融合转换~~ | ~~逐专家存储 → 融合格式的转换验证~~ | ✅ **已完成** |

### 4.4 实测得到的一个关键取舍：用 BF16 基底，不要用 -FP8

拉取两个变体的**官方权重索引**后（287,119 条 vs 1,609 条）发现：

| | `Qwen3.8-2.4T-A95B`（BF16） | `...-FP8` |
|---|---|---|
| 权重总数 | 1,609 | 287,119 |
| 专家存储 | **打包** `mlp.experts.gate_up_proj` / `down_proj` | **逐专家** `mlp.experts.N.{gate,up,down}_proj` |
| `weight_scale_inv` 块缩放 | 无 | 142,848 |

适配器通过 `_get_state_dict` 读 checkpoint：它**遍历模块自身的参数名**去权重索引里查，
查不到就**静默跳过**。而模块的参数名是**打包形态的** — 所以：

- **BF16 版匹配** → 专家权重能被正常找到与加载
- **FP8 版不匹配** → 专家权重全部找不到，随后的 `load_state_dict` 因缺键报错

因此**两个实践配置都应以 BF16 基底模型作为输入**。若确实要从 FP8 checkpoint 出发，需要先做
FP8 → BF16 的权重转换（上游对 DeepSeek 系有这条路径，Qwen3.8 未验证）。

> 附注：`Qwen3.8-27B` 的 BF16 与 FP8 变体都用 VLM 命名 `model.language_model.*`，与上游预期一致；
> 差异仅在 FP8 版多出 407 处 `weight_scale_inv`。

## 5. 快速开始

```bash
pip install -r requirements.txt

# 本地开发需要 msmodelslim.config 可见（setup.py 安装后会自动就位）
ln -sfn ../config msmodelslim/config

# 本分支新增的测试
python -m pytest test/cases/model/qwen3_5_moe_text/ -q
```

```bash
# Qwen3.8-2.4T-A95B（纯文本 MoE，W8A8）
# 用 BF16 基底，不要用 -FP8 变体（原因见 §4.3）
msmodelslim quant \
  --model_path <Qwen/Qwen3.8-2.4T-A95B 路径> \
  --save_path  <输出路径> \
  --config     lab_practice/qwen3_5_moe_text/qwen3_8_2_4t_a95b_w8a8.yaml

# Qwen3.8-27B（Dense VLM，W8A8）
msmodelslim quant \
  --model_path <Qwen/Qwen3.8-27B 路径> \
  --save_path  <输出路径> \
  --config     lab_practice/qwen3_5_moe/qwen3_8_27b_w8a8.yaml
```

## 5. 校准数据集

`lab_calib/qwen38_calib.jsonl` 由 `tools/build_calib_dataset.py` **确定性生成**（同 seed 输出可复现），
共 64 条、7 个领域，其中 **50% 在仓库内合成**（算术 CoT、代码任务、表格推理、指令跟随），
其余从仓库已收录的公开语料按固定种子重采样。

| 性质 | 值 |
|---|---|
| 唯一性 | 64/64，无重复 |
| 与上游 `mix_calib.jsonl` 的内容重叠 | **0 条** |
| 合成 CoT 中的算式 | 全部独立复算通过 |
| 输出字段 | 严格为加载器要求的 `inputs_pretokenized` |

逐域来源、池大小与抽样方式记录在 `lab_calib/qwen38_calib.provenance.json`，便于审计与复现。

```bash
python tools/build_calib_dataset.py --seed 20260824
```

## 6. 本分支的提交

```bash
git log --oneline d190c5c3..HEAD
```

| 提交 | 内容 |
|---|---|
| `[Feature]` | 扁平文本配置的布局适配（`config_shim.py`） |
| `[Feature]` | 文本主干别名（`model_shim.py`） |
| `[Fix]` | 别名绕过 `nn.Module` 子模块注册表 |
| `[Feature]` | 适配器子类 + `config.ini` 注册 |
| `[Feature]` | 两个 W8A8 实践配置 |
| `[Feature]` | 可复现的自建校准数据集 |

## 7. 许可证与致谢

- 许可证：[木兰宽松许可证 v2](./LICENSE)，与上游一致
- 上游项目：[MindStudio ModelSlim](https://gitcode.com/Ascend/msmodelslim) ｜ [昇腾社区](https://www.hiascend.com/cn/developer/software/mindstudio)
- 本分支保留上游全部版权声明与文件头；新增文件沿用同一许可证
