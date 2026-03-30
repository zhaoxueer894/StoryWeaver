# StoryWeaver — 项目规范遵从性报告

**课程:** COMP5423 自然语言处理  
**项目:** StoryWeaver — AI 驱动的最终幻想 VII 文本冒险  
**审查日期:** 2026-03-31

---

## 规范概述

规范定义了四个强制性步骤：(1) 数据准备、(2) 算法设计、(3) 系统实现和 (4) 性能评估。以下各部分将每项要求映射到具体代码实现。

---

## 步骤 1 — 数据准备

> *"收集和组织相关数据集…通过清洁文本噪声、分割剧情单元和标注叙事逻辑来预处理数据。"*

### 1.1 数据集收集

项目从 Kaggle 收集了真实的 FF7 脚本。原始文件 `data/raw/ff7-script.csv` 是主要的训练和生成源。十个额外的《最终幻想》脚本存档在 `data/external/archive/` 下（FF5、FF6、FF7 重制版、FF7 危机核心、FF7 Opera Omnia 变体、王者之剑、FF 世界），提供补充语料材料。

### 1.2 预处理管道 (`02_preprocess.py`)

**清洁文本噪声** — `clean_text()` 应用三个正则表达式传递：
- `re.sub(r"\s+", " ", text)` — 折叠多余的空格
- `re.sub(r"\[.*?\]", "", text)` — 移除 `[按钮提示]` 注释
- `re.sub(r"\(.*?\)", "", text)` — 移除 `(舞台指导)` 括号注释

清理后少于 2 个字符的行会被删除。

**分割剧情单元** — 每 8 行连续对话被分组为一个剧情单元：
```python
df["plot_unit_id"] = (df.index // 8) + 1
```
这产生了语义上连贯的块，可以独立检索、评分和用作生成上下文。

**标注叙事逻辑** — `label_narrative_type()` 基于角色身份和关键词模式分配四个标签之一（`action`、`dialogue`、`narration`、`system`）。这些标签在运行时用于过滤其叙事类型最符合玩家意图的剧情单元。

### 1.3 输出数据集

生成了五个处理后的文件：

| 文件 | 用途 |
|---|---|
| `ff7_cleaned_base.csv` | 在 `load_plot_unit()` 中按轮查找叙事类型 |
| `dialogue_dataset.csv` | 加载为 `FF7_DIALOGUE_CORPUS` 的对话语料库 |
| `plot_units.csv` | 用于故事生成和导航的核心语料 |
| `plot_consistency_samples.csv` | 用于一致性评分的参考文本 |
| `character_roles.csv` | 按场景频率排序的角色选择 |

**一致性样本质量** — 正样本是单个连贯的剧情单元；负样本故意配对至少 5 个位置分开的单元（`abs(j - i) >= 5`）以确保真实的不一致，使用 `random.seed(42)` 实现可重复性。

---

## 步骤 2 — 算法设计

> *"探索最先进的 NLP 方法，包括上下文感知文本生成、用户意图识别、剧情一致性检测和对话管理。"*

### 2.1 上下文感知文本生成 (`FF7ContextualGenerator`)

**模型:** 通过 HuggingFace `AutoModelForCausalLM` 加载的 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`。

生成器实现了四个不同的生成函数，每个都具有使用模型官方聊天模板（`apply_chat_template`）格式化的专用系统/用户提示结构：

| 方法 | 目的 | `max_new_tokens` |
|---|---|---|
| `generate()` | 每轮故事续写 | 180 |
| `generate_intro()` | 固定的开场叙述（故事背景） | 160 |
| `generate_options()` | 三个 A/B/C 玩家行动选择 | 150 |
| `generate_ending()` | 游戏风格条件结局 | 200 |

**上下文感知** 通过三种机制实现：
1. `global_story_history` 的最后 512 个字符被注入到每个生成提示中，标记为 `"Story so far:"`。
2. 一个评分记忆堆（容量 6，见 §2.4）通过提示中的 `"Memory:"` 展示最具叙事重要性的过去事件。
3. `_trim_to_sentence()` 在最后一个 `.!?` 字符处裁剪所有输出，防止中途截断，无论令牌预算如何。

**选项生成鲁棒性** — `generate_options()` 使用两次尝试重试循环。如果第一次尝试产生少于 3 个有效的不同选项（由于格式错误、模板占位符回显、NPC 特性或重复），则为第二次尝试在系统提示中追加针对性的纠正注释。`_normalize_option()` 强制执行一致的第二人称 `"You …"` 格式，并硬拒绝：括号伪影（`[action]`）、`You (CharacterName)` NPC 特性模式和元评论短语。

### 2.2 零样本用户意图识别 (`FF7IntentRecognizer`)

**模型:** 通过 HuggingFace `pipeline("zero-shot-classification")` 的 `facebook/bart-large-mnli`。

定义了五个意图标签：`talk`、`explore`、`move`、`interact`、`observe`。对每个玩家行动，`recognize_intent()` 返回：
- `intent` — 主导标签
- `confidence_score` — 主导标签的概率（用于评估指标）
- `all_intents` — 完整分数分布

识别的意图通过映射到叙事类型优先级列表直接引导剧情单元导航：

```
talk     → [dialogue, narration, action]
move     → [action, narration, dialogue]
explore  → [action, narration, dialogue]
interact → [action, dialogue, narration]
observe  → [action, narration, dialogue]
```

### 2.3 剧情一致性检测 (`FF7ConsistencyChecker`)

**模型:** 通过 `sentence_transformers` 的 `all-MiniLM-L6-v2`（Sentence-BERT）。

`compute_consistency(history, new_text)` 产生一个加权复合分数：

```
最终得分 = 余弦相似度(历史, 新文本) × 0.7
        + 平均(余弦相似度(新文本, 参考_i) 对于参考_i 在 references[:20]) × 0.3
```

0.7/0.3 权重优先考虑与特定会话运行历史的连贯性，同时奖励与规范 FF7 参考语料库的对齐（来自 `plot_consistency_samples.csv`）。该分数在每轮后计算并存储以供最终评估。

### 2.4 对话管理 — 记忆堆

一个有限的优先级队列（`heapq`，容量 6）存储最重要的过去事件。重要性通过以下方式评分：
- 文本长度标准化为 120 个字符
- 规范角色名称出现次数（Cloud、Sephiroth、Tifa、Aerith、Barret、Shinra、Avalanche、Midgar）

当堆超过容量时，最低分项会被驱逐，仅保留最具叙事重要性的记忆。这些被序列化为每个生成提示中的 `"Memory: …"`，为 LLM 提供超越滑动窗口的长程上下文。

### 2.5 语义剧情导航

`choose_next_plot_unit()` 实现两阶段选择：
1. **意图过滤候选池** — 其 `narrative_types` 符合玩家意图优先级列表的剧情单元被优先考虑；按角色存在进行过滤以保持角色连贯性。
2. **Sentence-BERT 排名** — 过滤池（上限 50 个前向单元）按玩家选择文本和故事历史最后 300 个字符组合的查询与候选文本之间的余弦相似度进行排名：
   ```python
   查询 = f"{玩家选择}. {最近上下文}"
   分数 = util.cos_sim(查询_嵌入, 候选_嵌入)[0]
   最佳_idx = int(分数.argmax())
   ```
   这重新使用已加载的 `all-MiniLM-L6-v2` 编码器，无需额外的模型成本。

不同的玩家选择产生不同的查询向量，导致同一候选池产生不同的选定单元 — 这是不同的选择导致不同叙事路径和不同结局的主要机制。

### 2.6 游戏风格条件结局

`conclude_story()` 使用 `collections.Counter` 对每轮意图标签构建游戏风格概档。主导意图映射到原型描述符：

| 意图 | 原型 |
|---|---|
| `talk` | "一位优先考虑对话和联盟建立的外交官" |
| `explore` | "一位寻求隐藏路径和揭示秘密的探险家" |
| `move` | "一位毫不犹豫地向前冲的果断战士" |
| `interact` | "一位直接应对每个障碍的问题解决者" |
| `observe` | "一位在行动前研究形势的谨慎战略家" |

该描述符与完整意图分布、最后 5 个选择和顶部记忆项一起传递给 `generate_ending()`，产生在游戏风格之间真正不同的叙事结论。

---

## 步骤 3 — 系统实现

> *"使用 Hugging Face Transformers、PyTorch 和 Gradio 等框架开发交互式文本冒险游戏系统。"*

### 3.1 使用的框架

| 框架 | 角色 |
|---|---|
| HuggingFace Transformers | TinyLlama 因果 LM、BART-MNLI 零样本管道 |
| PyTorch | 模型推理、CUDA/CPU 设备调度 |
| sentence-transformers | 用于一致性和导航的 Sentence-BERT 嵌入 |
| Gradio | 交互式 Web UI（`gr.Blocks`） |
| pandas | 所有数据集 I/O 和过滤操作 |

### 3.2 用户界面 (`04_app.py`)

Gradio `Blocks` 布局包含两个相互排斥的列：

**游戏列**（游戏过程中可见）：
- **故事背景** — 固定文本框，在角色确认时从 `generate_intro()` 设置一次；游戏过程中不会被覆盖
- **角色选择器** — 按场景频率排序的前 12 个角色的下拉菜单，通过按钮确认
- **故事** — 滚动文本框，在每个玩家行动后累积完整的会话历史
- **可用行动** — 每轮后由 LLM 重新生成的三个 A/B/C 选项
- **进度** — 实时计数器显示 `剩余互动: N / 10`
- **A / B / C 按钮** — 使用相应的选择索引触发 `player_action()`

**结局列**（隐藏直到第 10 轮）：
- **故事结局** — LLM 生成的游戏风格条件结论
- **性能评估** — 格式化的指标报告
- **重新开始游戏** — 调用 `game.reset()`，恢复所有状态而无需重新加载 ML 模型

### 3.3 实时生成管道

每次按钮点击时，系统按顺序执行：意图识别 → 语义剧情导航 → 故事生成 → 一致性评分 → 记忆更新 → 选项生成。每个步骤都记录到 `logs/storyweaver.log`，带有标记的部分（`[GENERATION PROMPT]`、`[GENERATION RESPONSE]`、`[OPTIONS PROMPT (attempt N)]`、`[NAVIGATION]` 等）。

### 3.4 会话管理

`FF7AdventureEngine.reset()` 就地重置所有会话状态（玩家角色、记忆堆、故事历史、指标列表、轮次计数器），因此三个加载的 ML 模型在游戏之间不会重新加载 — 这使重启保持近乎即时。

---

## 步骤 4 — 性能评估

> *"评估系统的叙事质量、交互响应性和用户体验…测量剧情连贯性分数、生成响应时间、玩家选择匹配准确性和沉浸式游戏体验的满意度。"*

### 4.1 收集的指标

| 指标 | 收集点 | 公式 |
|---|---|---|
| 剧情连贯性分数 | 每次 `generate()` 调用后 | Sentence-BERT 加权余弦（§2.3） |
| 生成响应时间 | `time.time()` 在 `generate()` 前/后 | 挂钟秒数 |
| 意图置信度 | 每次 `recognize_intent()` 调用后 | BART-MNLI 顶标签概率 |
| 沉浸分数 | 在 `conclude_story()` 中计算一次 | `coherence×0.5 + intent×0.3 + speed×0.2` |

沉浸分数的速度奖励是 `min(1.0, 1.0 / 平均_响应_时间)`，奖励更快的响应的更高贡献。

### 4.2 评估报告

`format_metrics()` 呈现一个结构化报告，在游戏结束后立即显示在结局 UI 中：

```
====================================================
           性能评估
====================================================

完成轮数          : 10

--- 叙事质量 ---
  剧情连贯性 (平均) : 0.XXXX  [好]
  每轮连贯性  : [0.XXXX, ...]

--- 交互响应性 ---
  平均响应时间    : X.XX 秒
  响应时间  : [Xs, ...]

--- 玩家选择匹配 ---
  意图置信度    : 0.XXXX  [优秀]
  所做选择         :
    第 1 轮: 你…
    ...

--- 沉浸式体验 ---
  沉浸分数      : 0.XXXX  [一般]
  (连贯性×0.5 + 意图×0.3 + 速度×0.2)
====================================================
```

质性评级（优秀 / 好 / 一般 / 需要改进）使用阈值应用于 `avg_plot_coherence`、`avg_intent_confidence` 和 `immersion_score`：≥0.80、≥0.65、≥0.50。

---

## 总结

| 规范要求 | 实现 |
|---|---|
| 收集的文本冒险游戏脚本 | `data/raw/ff7-script.csv` + 10 个外部 FF 脚本 |
| 分支叙事语料 | `plot_units.csv`（8 行段） |
| 对话数据集 | `dialogue_dataset.csv` |
| 剧情一致性注释样本 | `plot_consistency_samples.csv`（正样本 + 间隙≥5 的负对） |
| 清洁文本噪声 | `02_preprocess.py` 中的 `clean_text()` |
| 分割剧情单元 | `plot_unit_id = index // 8 + 1` |
| 标注叙事逻辑 | `label_narrative_type()` → 4 个标签 |
| 上下文感知文本生成 | TinyLlama-1.1B-Chat 与聊天模板、历史窗口、记忆提示 |
| 用户意图识别 | BART-large-MNLI 零样本、5 个意图 |
| 剧情一致性检测 | Sentence-BERT 余弦相似度、0.7/0.3 加权 |
| 对话管理 | 评分记忆堆（容量 6）、3072 字符滑动历史窗口 |
| 实时生成管道 | 每轮：意图 → 导航 → 生成 → 评分 → 选项 |
| 高效剧情分支 | 意图过滤 + Sentence-BERT 排名的 50 单元前向窗口 |
| 个性化结局 | 基于 `Counter` 的游戏风格分析 → 原型条件生成 |
| HuggingFace Transformers | TinyLlama、BART-MNLI、all-MiniLM-L6-v2 |
| PyTorch | 模型加载、CUDA/CPU 调度 |
| Gradio | `gr.Blocks` 双列游戏/结局布局 |
| 剧情连贯性分数 | 每轮 Sentence-BERT 分数 |
| 生成响应时间 | 每次 `generate()` 调用的挂钟计时 |
| 玩家选择匹配准确性 | 每轮 BART-MNLI 置信度分数 |
| 沉浸式体验指标 | 复合分数：连贯性×0.5 + 意图×0.3 + 速度×0.2 |
