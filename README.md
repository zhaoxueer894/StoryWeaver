# StoryWeaver

StoryWeaver 是一个基于《最终幻想 VII》文本冒险的自然语言处理项目。它使用游戏脚本数据，构建一个数据驱动的剧情生成与一致性评估系统，并通过 Gradio 提供可交互的文本冒险界面。

## 项目目标

- 数据清洗与剧情分段
- NLP 模型驱动的互动剧情生成
- 零样本玩家意图识别
- 剧情一致性检测与评价
- 通过 Gradio 展示可共享文本冒险体验

## 目录结构

- `data/raw/`：原始剧本文本数据
- `data/processed/`：预处理后可直接使用的 CSV 数据集
- `scripts/02_preprocess.py`：数据清洗与特征工程
- `scripts/core_algorithms.py`：核心模型与游戏引擎实现
- `scripts/04_app.py`：Gradio 可视化展示入口
- `README.md`：项目介绍文档

## 依赖环境

推荐安装：

```bash
pip install gradio pandas torch transformers sentence-transformers huggingface-hub
```

如果需要，你也可以建立 conda 环境并安装上述包。

## 一步一步处理逻辑

### 1. 数据预处理：`scripts/02_preprocess.py`

该脚本将原始 `data/raw/ff7-script.csv` 处理成 4 个用于后续建模的标准文件：

1. 读取原始脚本并查看数据结构
2. 清洗文本噪声
   - 去除多余空格
   - 删除方括号 `[...]` 和圆括号 `(...)` 中的注释
   - 过滤空行和过短文本
3. 将连续对话按固定长度划分为剧情单元
   - 这里采用每 8 句对话为一个 `plot_unit_id`
4. 自动标注叙事类型
   - `action`：包含移动、开启、进入等动作词
   - `dialogue`：包含问句、对话词汇
   - `narration`：叙事类文本
   - `system`：系统提示或旁白类
5. 输出四个 processed 文件：
   - `ff7_cleaned_base.csv`：核心清洗数据集
   - `dialogue_dataset.csv`：对话语料，适合意图识别与生成
   - `plot_units.csv`：剧情单元数据，适合剧情生成与分支控制
   - `plot_consistency_samples.csv`：一致性样本，适合后续连贯性评分

### 2. 核心模型与游戏引擎：`scripts/core_algorithms.py`

该文件实现四个模块：

#### 2.1 FF7ContextualGenerator

- 使用 `distilgpt2` 进行文本生成
- 以当前剧情历史、玩家选择、当前角色为 prompt 上下文
- 生成连贯的续写文本

#### 2.2 FF7IntentRecognizer

- 使用 `facebook/bart-large-mnli` 的零样本分类
- 将玩家选项映射到意图标签：`talk`, `explore`, `move`, `interact`, `observe`
- 输出意图类别与置信度

#### 2.3 FF7ConsistencyChecker

- 使用 `SentenceTransformer('all-MiniLM-L6-v2')`
- 计算生成文本与历史上下文之间的语义相似度
- 同时与一组一致性样本进行对比，融合两个相似度得出最终一致性分数

#### 2.4 FF7AdventureEngine

这是项目的核心游戏逻辑：

- 初始化时加载三大模型和四个 processed 数据集
- 维持全局剧情历史 `global_story_history`
- 加载当前剧情单元并生成玩家可选动作
- 玩家选择后执行完整流程：
  1. 意图识别
  2. 上下文剧情生成
  3. 剧情一致性评分
  4. 更新全局记忆
  5. 切换到下一个剧情单元

该引擎通过 `build_action_options()` 自动生成三个按钮选项，确保每次交互都有可用行为。

## 3. 可视化展示：`scripts/04_app.py`

该脚本基于 Gradio 构建一个文本冒险界面：

- 当前剧情场景显示
- 玩家动作输出
- 意图与一致性评分显示
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
