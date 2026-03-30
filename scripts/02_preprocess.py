import os
import re
import pandas as pd

# ======================
# 路径配置（完全匹配你的项目）
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "ff7-script.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ======================
# 1. 读取原始数据（你真实的格式）
# ======================
df = pd.read_csv(RAW_PATH)
print(f"原始数据条数: {len(df)}")
print("数据列名:", df.columns.tolist())

# ======================
# 2. 清洗文本噪声（任务书要求：cleaning text noise）
# ======================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)          # 多余空格
    text = re.sub(r"\[.*?\]", "", text)       # 去掉 [操作提示]
    text = re.sub(r"\(.*?\)", "", text)       # 去掉 (注释)
    text = text.strip()
    return text

df["clean_dialogue"] = df["Dialogue"].apply(clean_text)

# 过滤无效数据
df = df[df["clean_dialogue"].str.len() >= 2]
df = df[~df["clean_dialogue"].isna()]
df = df[df["clean_dialogue"] != ""]

print(f"清洗后数据条数: {len(df)}")

# ======================
# 3. 分段剧情单元（任务书要求：segmenting plot units）
# 按连续对话顺序生成 plot_unit（适合后续剧情生成 & 一致性校验）
# ======================
df = df.reset_index(drop=True)
df["plot_unit_id"] = (df.index // 8) + 1  # 每8句对话为一个剧情单元

# ======================
# 4. 标注叙事逻辑（任务书要求：labeling narrative logic）
# 用于后续量化评价、分类、一致性判断
# 标签：action / dialogue / narration / system
# ======================
def label_narrative_type(dialogue, character):
    dialogue = dialogue.lower()
    char = str(character).lower()

    if char in ["on-screen", "system", "narrator"]:
        return "system"
    elif any(w in dialogue for w in ["go", "follow", "run", "push", "open", "enter", "wait"]):
        return "action"
    elif any(w in dialogue for w in ["ask", "tell", "say", "why", "what", "hello"]):
        return "dialogue"
    else:
        return "narration"

df["narrative_type"] = df.apply(
    lambda row: label_narrative_type(row["clean_dialogue"], row["Character"]), axis=1
)

# ======================
# 以下输出 4 个标准化 processed 文件
# 全部可直接用于后续：训练、生成、量化评价、一致性校验
# ======================

# ------------------------------------------------------------------------------------------
# 输出 1：基础清洗数据集（主文件）
# ------------------------------------------------------------------------------------------
base_clean = df[["Character", "clean_dialogue", "narrative_type", "plot_unit_id"]].copy()
base_clean.to_csv(
    os.path.join(PROCESSED_DIR, "ff7_cleaned_base.csv"),
    index=False, encoding="utf-8-sig"
)

# ------------------------------------------------------------------------------------------
# 输出 2：对话数据集（dialogue datasets）→ 可直接用于意图识别 & 对话生成
# ------------------------------------------------------------------------------------------
dialogue_df = df[["Character", "clean_dialogue", "narrative_type"]].copy()
dialogue_df = dialogue_df.rename(columns={"clean_dialogue": "dialogue"})
dialogue_df.to_csv(
    os.path.join(PROCESSED_DIR, "dialogue_dataset.csv"),
    index=False, encoding="utf-8-sig"
)

# ------------------------------------------------------------------------------------------
# 输出 3：剧情单元数据集（branching narrative corpora）→ 可直接用于剧情生成 & 连贯性评估
# ------------------------------------------------------------------------------------------
plot_units = []
for pid, group in df.groupby("plot_unit_id"):
    plot_units.append({
        "plot_unit_id": pid,
        "character_list": ", ".join(group["Character"].unique()),
        "dialogue_count": len(group),
        "narrative_types": ", ".join(group["narrative_type"].unique()),
        "full_plot_text": "\n".join(group["clean_dialogue"])
    })

plot_df = pd.DataFrame(plot_units)
plot_df.to_csv(
    os.path.join(PROCESSED_DIR, "plot_units.csv"),
    index=False, encoding="utf-8-sig"
)

# ------------------------------------------------------------------------------------------
# 输出 4：角色候选列表（role candidates）
# → 启动时从数据集选择角色
# ------------------------------------------------------------------------------------------
role_records = []
for _, row in plot_df.iterrows():
    chars = [c.strip() for c in str(row["character_list"]).split(",") if c.strip()]
    for char in chars:
        role_records.append({
            "character": char,
            "plot_unit_id": row["plot_unit_id"]
        })
role_df = pd.DataFrame(role_records)
role_summary = role_df.groupby("character")["plot_unit_id"].agg(list).reset_index()
role_summary["plot_unit_count"] = role_summary["plot_unit_id"].apply(len)
role_summary["plot_unit_ids"] = role_summary["plot_unit_id"].apply(lambda ids: ";".join(map(str, sorted(set(ids)))))
role_summary = role_summary[["character", "plot_unit_count", "plot_unit_ids"]]
role_summary.to_csv(
    os.path.join(PROCESSED_DIR, "character_roles.csv"),
    index=False, encoding="utf-8-sig"
)

# ------------------------------------------------------------------------------------------
# 输出 5：剧情一致性标注样本（plot consistency annotation samples）
# → 可直接用于量化评价（consistency score / coherence test）
# ------------------------------------------------------------------------------------------
consistency_samples = []
plot_list = plot_df.to_dict("records")

import random
random.seed(42)

for i, p in enumerate(plot_list):
    # Positive sample: a single coherent plot unit
    consistency_samples.append({
        "sample_id": len(consistency_samples) + 1,
        "plot_text": p["full_plot_text"],
        "is_consistent": 1
    })
    # Negative sample: pair with a non-adjacent unit (gap >= 5) to ensure true incoherence.
    far_indices = [j for j in range(len(plot_list)) if abs(j - i) >= 5]
    if far_indices:
        j = random.choice(far_indices)
        mixed = plot_list[j]
        consistency_samples.append({
            "sample_id": len(consistency_samples) + 1,
            "plot_text": p["full_plot_text"] + "\n---\n" + mixed["full_plot_text"],
            "is_consistent": 0
        })

consistency_df = pd.DataFrame(consistency_samples)
consistency_df.to_csv(
    os.path.join(PROCESSED_DIR, "plot_consistency_samples.csv"),
    index=False, encoding="utf-8-sig"
)

# ======================
# 最终输出
# ======================
print("\n✅ 数据处理完成！")
print(f"📁 输出目录：{PROCESSED_DIR}")
print(f"• ff7_cleaned_base.csv      → 清洗后主数据")
print(f"• dialogue_dataset.csv      → 对话语料（意图识别/生成）")
print(f"• plot_units.csv            → 剧情单元（剧情生成）")
print(f"• plot_consistency_samples.csv → 一致性样本（量化评价）")