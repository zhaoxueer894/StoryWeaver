# StoryWeaver — AI-Powered Final Fantasy VII Text Adventure

一个基于《最终幻想 VII》游戏脚本的 NLP 驱动交互式文本冒险游戏。本项目整合多个深度学习模型实现数据驱动的剧情生成、玩家意图识别、剧情一致性评估，并通过 Gradio 提供实时交互界面。

**课程:** COMP5423 自然语言处理  
**开发日期:** 2026年1月-3月  
**语言:** Python 3.8+

---

## 项目特色

✨ **上下文感知文本生成** — TinyLlama 1.1B Chat 模型，集成滑动窗口历史与记忆堆  
🎯 **零样本意图识别** — BART-MNLI 自动识别玩家意图（谈话、探索、移动等）  
🔗 **语义剧情一致性检测** — Sentence-BERT 加权融合计算生成文本与历史连贯性  
📖 **动态剧情路由** — Intent + Sentence-BERT 排名选择下一个剧情单元  
🎮 **风格条件结局** — 根据玩家 10 轮选择的意图分布生成个性化结局  
📊 **实时性能评估** — 自动收集一致性、意图置信度、响应时间、沉浸感评分

---

## 项目结构

```
StoryWeaver/
├── data/
│   ├── raw/                              # 原始 FF7 脚本
│   │   └── ff7-script.csv
│   ├── external/
│   │   └── archive/                      # 补充语料：FF5, FF6, FF7 Remake 等
│   └── processed/                        # 预处理输出
│       ├── ff7_cleaned_base.csv          # 清洗后的基础数据
│       ├── dialogue_dataset.csv          # 对话语料库
│       ├── plot_units.csv                # 剧情单元（核心）
│       ├── plot_consistency_samples.csv  # 一致性参考样本
│       └── character_roles.csv           # 角色频率排序
├── scripts/
│   ├── 01_download_data.py               # 数据下载脚本
│   ├── 02_preprocess.py                  # 数据清洗与特征工程
│   ├── core_algorithms.py                # 核心 NLP 模块与游戏引擎
│   └── 04_app.py                         # Gradio Web UI 入口
├── logs/
│   └── storyweaver.log                   # 运行时日志
├── requirements.txt                      # Python 依赖（pip）
├── environment.yml                       # Conda 环境配置
├── report.md                             # 规范遵从性详细报告
├── specification.md                      # 项目规范文档
└── README.md                             # 本文件
```

---

## 快速开始

### 1. 环境配置

#### 方案 A：Conda（推荐）
```bash
conda env create -f environment.yml
conda activate storyweaver
```

#### 方案 B：pip + venv
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 数据预处理

```bash
cd scripts
python 02_preprocess.py
```

输出文件保存至 `data/processed/`：
- `ff7_cleaned_base.csv` — 去噪后的完整对话
- `dialogue_dataset.csv` — 精选对话语料
- `plot_units.csv` — 每 8 行分段的剧情单元
- `plot_consistency_samples.csv` — 正/负对样本（用于一致性评分参考）
- `character_roles.csv` — 按出现频率排序的角色列表

### 3. 启动交互式游戏

```bash
python 04_app.py
```

打开浏览器，访问 `http://localhost:7860` 开始游戏。

---

## 核心算法

### FF7ContextualGenerator（剧情生成）

**模型:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**输入:** 
- 最近 512 字符的剧情历史
- 当前场景描述
- 玩家的选择
- 重要事件记忆（堆大小 6）

**输出:** 2-3 句连贯的 FF7 故事续写

**特性:**
- 官方 Chat 模板确保输出格式一致
- `_trim_to_sentence()` 防止中途截断
- 生成 4 个不同函数：`generate()` / `generate_intro()` / `generate_options()` / `generate_ending()`
- 选项生成带重试验证（确保 3 个不同的有效选项）

### FF7IntentRecognizer（意图识别）

**模型:** `facebook/bart-large-mnli` (Zero-shot Classification)  
**意图标签:** 
- `talk` — 交谈、对话
- `explore` — 探索、观察
- `move` — 移动、冲锋
- `interact` — 互动、操纵
- `observe` — 监视、思考

**输出:** 意图类别 + 置信度 (0-1)

**用途:** 
- 直接影响剧情单元导航优先级
- 用于结局代码生成和沉浸感评分

### FF7ConsistencyChecker（一致性评分）

**模型:** `all-MiniLM-L6-v2` (Sentence-BERT)  
**公式:**
```
final_score = cos_sim(history, new_text) × 0.7
            + mean(cos_sim(new_text, reference_i) for i in top_20) × 0.3
```

**解释:**
- 70% 权重：新生成文本与本轮剧情历史的连贯性
- 30% 权重：与 FF7 官方语料库 20 条最相关参考的对齐度

**输出:** 0-1 连贯性分数

### FF7AdventureEngine（游戏核心逻辑）

**状态维护:**
- `global_story_history` — 全轮次累积故事
- `memory_heap` — 最重要事件的有限优先队列（容量 6）
- `current_plot_unit_id` — 当前剧情位置
- `round_counter` — 交互轮次（最多 10 轮）

**单轮流程:**
1. **意图识别** — 解析玩家选择的语义
2. **剧情导航** — Intent 过滤 + Sentence-BERT 排名选择下一单元
3. **故事生成** — TinyLlama 产生 2-3 句续写
4. **一致性评分** — Sentence-BERT 计算本轮连贯性
5. **记忆更新** — 按重要性维护 6 项事件堆
6. **选项生成** — LLM 产生 A/B/C 三个新行动

**结局生成（第 10 轮后）:**
- 统计 10 轮意图分布，确定玩家风格
- 根据风格生成对应原型描述
- TinyLlama 产生 4-5 句的个性化结局

### 记忆堆（Dialogue Management）

```python
# 重要性评分 = normalize(text_len, 120) + char_occurrence_count
# 规范字符: Cloud, Sephiroth, Tifa, Aerith, Barret, Shinra, Avalanche, Midgar
# 容量: 6 条
# 当超过容量时，自动驱逐最低分项
# 在每轮 generation prompt 中的 "Memory:" 部分注入
```

---

## Gradio UI 设计

### 游戏列 (gameplay column)
- **故事背景** — 游戏开始时由 `generate_intro()` 产生的固定背景
- **角色选择** — 下拉菜单（按出现频率排序的前 12 个角色）
- **故事文本** — 实时累积的全轮故事（只读文本框）
- **可用行动** — 实时生成的 A/B/C 三个选项按钮
- **进度显示** — "剩余互动: X / 10"
- **A / B / C 按钮** — 触发 `player_action(choice_idx)`

### 结局列 (ending column，第 10 轮后显示)
- **故事结局** — LLM 生成的代码化结局
- **性能评估** — 格式化指标报告
- **重新开始** — `game.reset()` 恢复状态，模型无需重载

---

## 性能指标

| 指标 | 收集点 | 公式 / 说明 |
|---|---|---|
| **Plot Coherence** | 每轮 `generate()` 后 | Sentence-BERT 加权余弦相似度 |
| **Response Time** | 每轮生成前后计时 | wall-clock 秒数 |
| **Intent Confidence** | 每轮 `recognize_intent()` 后 | BART-MNLI 置信度 (0-1) |
| **Immersion Score** | 结局时计算一次 | 0.5×coherence + 0.3×intent + 0.2×speed |

**质量分级阈值:**
- ≥0.80: Excellent
- ≥0.65: Good
- ≥0.50: Fair
- <0.50: Needs Improvement

---

## 依赖库

详见 `requirements.txt` 和 `environment.yml`

主要库:
- **gradio** ≥3.35 — Web UI
- **pandas** ≥1.5 — 数据处理
- **torch** ≥2.0 — PyTorch 深度学习框架
- **transformers** ≥4.30 — HuggingFace 模型加载
- **sentence-transformers** ≥2.2 — Sentence-BERT 嵌入
- **huggingface-hub** — 模型下载缓存

---

## 日志与调试

所有运行时日志保存至 `logs/storyweaver.log`:

```
[GENERATION PROMPT]  — 发送给 TinyLlama 的完整 prompt
[GENERATION RESPONSE]  — 模型原始输出
[OPTIONS PROMPT (attempt N)]  — 选项生成的 prompt（可能重试）
[OPTIONS RESPONSE (attempt N)]  — 选项生成的原始响应
[NAVIGATION]  — 剧情单元选择日志
[INTENT RECOGNITION]  — 意图识别结果
[CONSISTENCY SCORE]  — 一致性评分详情
```

---

## 性能参数调优

在 `core_algorithms.py` 中可调整：

- **温度 (temperature)** — 0.5 (generation) / 0.6 (intro) / 0.7 (options) / 0.6–0.8 (ending)
- **Top-P** — 0.9 (nucleus sampling)
- **Repetition Penalty** — 1.2–1.3 (减少重复)
- **Max New Tokens** — 180 (generation) / 160 (intro) / 150 (options) / 200 (ending)

---

## 常见问题

**Q: 模型太慢？**  
A: TinyLlama 是轻量级模型；首次运行会下载权重（~3GB）。后续启动即时。可使用 GPU 加速。

**Q: CUDA 显存不足？**  
A: 代码自动回退到 CPU。或在 `core_algorithms.py` 第 53 行调整 `torch_dtype` 为 `bfloat16` 以节省显存。

**Q: 结局千篇一律？**  
A: 检查 `player_action()` 中意图计数器是否正确更新。加大温度参数提高多样性。

**Q: 日志文件丢失？**  
A: 确保 `logs/` 目录存在。首次运行时自动创建，之后覆盖写入。

---

## 项目相关文件

- **`report.md`** — 详细规范遵从性报告，逐项映射每个规范要求到代码实现
- **`specification.md`** — 原始项目规范文档（4 步要求）
- **`logs/storyweaver.log`** — 运行时日志，用于调试

---

## 贡献与改进方向

🔮 **计划优化:**
- [ ] 支持多语言生成
- [ ] 流式 UI（实时显示生成过程）
- [ ] 存档/加载游戏进度
- [ ] 玩家游戏风格个性化推荐
- [ ] A/B 测试不同模型（Qwen, LLaMA 等）

---

## 许可证

本项目为 COMP5423 课程作业。代码、数据均为教学用途。

---

**最后更新:** 2026-03-31
- 生成剧情结果显示
- 最近故事历史显示
- 3 个按钮对应当前可选动作

启动时会创建 `FF7AdventureEngine`，并将按钮点击绑定到 `handle_player_choice()`。

如果使用 `share=True` 启动，Gradio 会在终端输出一个 `public_url`，你可以将该链接分享给其他人访问。

## 如何运行

1. 如果还没做预处理：

```bash
python scripts/02_preprocess.py
```

2. 启动交互界面：

```bash
python scripts/04_app.py
```

3. 终端会显示：

- 本地地址，如 `http://127.0.0.1:7860`
- 公网地址，如 `https://xxxxxxx.gradio.app`

将 `public_url` 复制给别人即可让他人访问。

## 日志与样本测试

- 运行 `scripts/04_app.py` 时，会自动在 `logs/storyweaver.log` 中生成新的运行日志。
- 每次启动前会删除旧日志，保证日志文件只包含本次运行内容。
- 日志中会记录：模型名称、模型参数、当前角色、玩家选择、实际发送的 prompt、生成结果、历史记忆等详细信息。
- 简单测试样本已放在 `data/samples/`：
  - `test_interactions.csv`
  - `test_intents.csv`
- 可以运行 `scripts/05_test_samples.py` 进行样本测试。

## 关键设计点

- 数据驱动：通过 `plot_units.csv` 与 `ff7_cleaned_base.csv` 生成剧情和选项
- 上下文感知：文本生成时保留全局历史，避免失去前因
- 零样本意图识别：无需额外标注即可识别玩家行动目的
- 剧情一致性评分：结合上下文相似度与标准一致性样本，量化生成质量

## 注意事项

- 运行本项目前请确保 `data/raw/ff7-script.csv` 存在且格式正确
- 预处理结果存储在 `data/processed/`
- 若使用 GPU，需要正确安装 `torch` 并确保 CUDA 可用
