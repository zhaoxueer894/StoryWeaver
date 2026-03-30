# StoryWeaver — Project Specification Compliance Report

**Course:** COMP5423 Natural Language Processing
**Project:** StoryWeaver — AI-Powered Final Fantasy VII Text Adventure
**Review Date:** 2026-03-31

---

## Specification Overview

The specification defines four mandatory steps: (1) Data Preparation, (2) Algorithm Design, (3) System Implementation, and (4) Performance Evaluation. The following sections map each requirement to the concrete code that addresses it.

---

## Step 1 — Data Preparation

> *"Collect and organize relevant datasets… Preprocess the data by cleaning text noise, segmenting plot units, and labeling narrative logic."*

### 1.1 Dataset Collection

The project collects an authentic FF7 script from Kaggle. The raw file `data/raw/ff7-script.csv` is the primary training and generation source. Ten additional Final Fantasy scripts are archived under `data/external/archive/` (FF5, FF6, FF7 Remake, FF7 Crisis Core, FF7 Opera Omnia variants, Kingsglaive, World of FF), providing supplementary corpus material.

### 1.2 Preprocessing Pipeline (`02_preprocess.py`)

**Cleaning text noise** — `clean_text()` applies three regex passes:
- `re.sub(r"\s+", " ", text)` — collapses redundant whitespace
- `re.sub(r"\[.*?\]", "", text)` — removes `[button prompt]` annotations
- `re.sub(r"\(.*?\)", "", text)` — removes `(stage direction)` parenthetical comments

Rows with fewer than 2 characters after cleaning are dropped.

**Segmenting plot units** — every 8 consecutive dialogue lines are grouped into one plot unit:
```python
df["plot_unit_id"] = (df.index // 8) + 1
```
This produces semantically coherent chunks that can be independently retrieved, scored, and used as generation context.

**Labeling narrative logic** — `label_narrative_type()` assigns one of four labels (`action`, `dialogue`, `narration`, `system`) based on character identity and keyword patterns. These labels are used at runtime to filter plot units whose narrative type best matches the player's intent.

### 1.3 Output Datasets

Five processed files are produced:

| File | Used for |
|---|---|
| `ff7_cleaned_base.csv` | Per-round narrative type lookup in `load_plot_unit()` |
| `dialogue_dataset.csv` | Dialogue corpus loaded as `FF7_DIALOGUE_CORPUS` |
| `plot_units.csv` | Core corpus for story generation and navigation |
| `plot_consistency_samples.csv` | Reference texts for consistency scoring |
| `character_roles.csv` | Role selection sorted by scene frequency |

**Quality of consistency samples** — positive samples are single coherent plot units; negative samples deliberately pair units at least 5 positions apart (`abs(j - i) >= 5`) to ensure genuine incoherence, with `random.seed(42)` for reproducibility.

---

## Step 2 — Algorithm Design

> *"Explore state-of-the-art NLP approaches including context-aware text generation, user intent recognition, plot consistency detection, and dialogue management."*

### 2.1 Context-Aware Text Generation (`FF7ContextualGenerator`)

**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` loaded via HuggingFace `AutoModelForCausalLM`.

The generator implements four distinct generation functions, each with a dedicated system/user prompt structure formatted with the model's official chat template (`apply_chat_template`):

| Method | Purpose | `max_new_tokens` |
|---|---|---|
| `generate()` | Per-round story continuation | 180 |
| `generate_intro()` | Fixed opening narrative (story background) | 160 |
| `generate_options()` | Three A/B/C player action choices | 150 |
| `generate_ending()` | Play-style-conditioned conclusion | 200 |

**Context awareness** is achieved through three mechanisms:
1. The last 512 characters of `global_story_history` are injected into every generation prompt under `"Story so far:"`.
2. A scored memory heap (capacity 6, described in §2.4) surfaces the most narratively significant past events via `"Memory:"` in the prompt.
3. `_trim_to_sentence()` clips all output at the last `.!?` character, preventing mid-sentence truncation regardless of the token budget.

**Option generation robustness** — `generate_options()` uses a two-attempt retry loop. If attempt 1 produces fewer than 3 valid distinct options (due to wrong format, template placeholder echo, NPC attribution, or duplicates), a targeted correction note is appended to the system prompt for attempt 2. `_normalize_option()` enforces a uniform second-person `"You …"` format and hard-rejects: bracket artifacts (`[action]`), `You (CharacterName)` NPC attribution patterns, and meta-commentary phrases.

### 2.2 Zero-Shot User Intent Recognition (`FF7IntentRecognizer`)

**Model:** `facebook/bart-large-mnli` via HuggingFace `pipeline("zero-shot-classification")`.

Five intent labels are defined: `talk`, `explore`, `move`, `interact`, `observe`. For every player action, `recognize_intent()` returns:
- `intent` — dominant label
- `confidence_score` — probability of the dominant label (used in evaluation metrics)
- `all_intents` — full score distribution

The recognised intent directly steers plot unit navigation by mapping to a narrative type priority list:

```
talk     → [dialogue, narration, action]
move     → [action, narration, dialogue]
explore  → [action, narration, dialogue]
interact → [action, dialogue, narration]
observe  → [action, narration, dialogue]
```

### 2.3 Plot Consistency Detection (`FF7ConsistencyChecker`)

**Model:** `all-MiniLM-L6-v2` (Sentence-BERT) via `sentence_transformers`.

`compute_consistency(history, new_text)` produces a weighted composite score:

```
final_score = cosine_sim(history, new_text) × 0.7
            + mean(cosine_sim(new_text, ref_i) for ref_i in references[:20]) × 0.3
```

The 0.7/0.3 weighting prioritises continuity with the specific session's running history while also rewarding alignment with the canonical FF7 reference corpus (sourced from `plot_consistency_samples.csv`). This score is computed after every round and stored for final evaluation.

### 2.4 Dialogue Management — Memory Heap

A bounded priority queue (`heapq`, capacity 6) stores the most important past events. Importance is scored by:
- Text length normalised to 120 characters
- Count of canonical character name occurrences (Cloud, Sephiroth, Tifa, Aerith, Barret, Shinra, Avalanche, Midgar)

When the heap exceeds capacity, the lowest-scoring entry is evicted, retaining only the most narratively significant memories. These are serialised into `"Memory: …"` in every generation prompt, giving the LLM long-range context beyond the sliding window.

### 2.5 Semantic Plot Navigation

`choose_next_plot_unit()` implements a two-stage selection:
1. **Intent-filtered candidate pool** — plot units whose `narrative_types` match the player's intent priority list are preferred; filtered by character presence to maintain role continuity.
2. **Sentence-BERT ranking** — the filtered pool (capped at 50 forward units) is ranked by cosine similarity between the candidate texts and a query combining the player's choice text with the last 300 characters of story history:
   ```python
   query = f"{player_choice}. {recent_ctx}"
   scores = util.cos_sim(query_emb, candidate_embs)[0]
   best_idx = int(scores.argmax())
   ```
   This reuses the already-loaded `all-MiniLM-L6-v2` encoder with no additional model cost.

Different player choices produce different query vectors, causing the same candidate pool to yield different selected units — this is the primary mechanism by which different choices lead to different narrative paths and different endings.

### 2.6 Play-Style-Conditioned Ending

`conclude_story()` builds a play-style profile using `collections.Counter` over the per-round intent labels. The dominant intent maps to an archetype descriptor:

| Intent | Archetype |
|---|---|
| `talk` | "a diplomat who prioritised conversation and alliance-building" |
| `explore` | "an explorer who sought hidden paths and uncovered secrets" |
| `move` | "a decisive warrior who pressed forward without hesitation" |
| `interact` | "a problem-solver who engaged directly with every obstacle" |
| `observe` | "a careful strategist who studied situations before acting" |

This descriptor, together with the full intent distribution, the last 5 choices, and the top memory items, is passed to `generate_ending()`, producing a narrative conclusion that genuinely differs between play-styles.

---

## Step 3 — System Implementation

> *"Develop the interactive text adventure game system using frameworks such as Hugging Face Transformers, PyTorch, and Gradio."*

### 3.1 Frameworks Used

| Framework | Role |
|---|---|
| HuggingFace Transformers | TinyLlama causal LM, BART-MNLI zero-shot pipeline |
| PyTorch | Model inference, CUDA/CPU device dispatch |
| sentence-transformers | Sentence-BERT embeddings for consistency and navigation |
| Gradio | Interactive web UI (`gr.Blocks`) |
| pandas | All dataset I/O and filtering operations |

### 3.2 User Interface (`04_app.py`)

The Gradio `Blocks` layout contains two mutually exclusive columns:

**Game Column** (visible during play):
- **Story Background** — pinned textbox set once on role confirmation from `generate_intro()`; never overwritten during gameplay
- **Role selector** — dropdown of the top-12 characters sorted by scene frequency, confirmed with a button
- **Story** — scrolling textbox that accumulates the full session history after each player action
- **Available Actions** — three A/B/C options regenerated by the LLM after each round
- **Progress** — live counter showing `Remaining interactions: N / 10`
- **A / B / C buttons** — trigger `player_action()` with the corresponding choice index

**Ending Column** (hidden until round 10):
- **Story Ending** — the LLM-generated play-style-conditioned conclusion
- **Performance Evaluation** — formatted metrics report
- **Restart Game** — calls `game.reset()`, restoring all state without reloading ML models

### 3.3 Real-Time Generation Pipeline

On each button click, the system executes in sequence: intent recognition → semantic plot navigation → story generation → consistency scoring → memory update → option generation. Each step is logged to `logs/storyweaver.log` with labelled sections (`[GENERATION PROMPT]`, `[GENERATION RESPONSE]`, `[OPTIONS PROMPT (attempt N)]`, `[NAVIGATION]`, etc.).

### 3.4 Session Management

`FF7AdventureEngine.reset()` resets all session state (player role, memory heap, story history, metric lists, round counter) in place, so the three loaded ML models are never reloaded between games — this keeps restart near-instant.

---

## Step 4 — Performance Evaluation

> *"Assess the system's narrative quality, interaction responsiveness, and user experience… Measure plot coherence scores, generation response times, player choice matching accuracy, and the satisfaction of immersive gaming experience."*

### 4.1 Metrics Collected

| Metric | Collection point | Formula |
|---|---|---|
| Plot coherence score | After each `generate()` call | Sentence-BERT weighted cosine (§2.3) |
| Generation response time | `time.time()` before/after `generate()` | Wall-clock seconds |
| Intent confidence | After each `recognize_intent()` call | BART-MNLI top-label probability |
| Immersion score | Computed once in `conclude_story()` | `coherence×0.5 + intent×0.3 + speed×0.2` |

The speed bonus for the immersion score is `min(1.0, 1.0 / avg_response_time)`, rewarding faster responses with a higher contribution.

### 4.2 Evaluation Report

`format_metrics()` renders a structured report displayed in the ending UI immediately after the game concludes:

```
====================================================
           PERFORMANCE EVALUATION
====================================================

Rounds Played          : 10

--- Narrative Quality ---
  Plot Coherence (avg) : 0.XXXX  [Good]
  Coherence per Round  : [0.XXXX, ...]

--- Interaction Responsiveness ---
  Avg Response Time    : X.XX s
  Response Times       : [Xs, ...]

--- Player Choice Matching ---
  Intent Confidence    : 0.XXXX  [Excellent]
  Choices Made         :
    Round 1: You …
    ...

--- Immersive Experience ---
  Immersion Score      : 0.XXXX  [Fair]
  (Coherence×0.5 + Intent×0.3 + Speed×0.2)
====================================================
```

Qualitative ratings (Excellent / Good / Fair / Needs Improvement) are applied to `avg_plot_coherence`, `avg_intent_confidence`, and `immersion_score` using thresholds: ≥0.80, ≥0.65, ≥0.50.

---

## Summary

| Specification requirement | Implementation |
|---|---|
| Text adventure game scripts collected | `data/raw/ff7-script.csv` + 10 external FF scripts |
| Branching narrative corpora | `plot_units.csv` (8-line segments) |
| Dialogue datasets | `dialogue_dataset.csv` |
| Plot consistency annotation samples | `plot_consistency_samples.csv` (positive + gap-≥5 negative pairs) |
| Cleaning text noise | `clean_text()` in `02_preprocess.py` |
| Segmenting plot units | `plot_unit_id = index // 8 + 1` |
| Labeling narrative logic | `label_narrative_type()` → 4 labels |
| Context-aware text generation | TinyLlama-1.1B-Chat with chat template, history window, memory prompt |
| User intent recognition | BART-large-MNLI zero-shot, 5 intents |
| Plot consistency detection | Sentence-BERT cosine similarity, 0.7/0.3 weighted |
| Dialogue management | Scored memory heap (capacity 6), 3072-char sliding history window |
| Real-time generation pipeline | Per-round: intent → navigation → generate → score → options |
| Efficient plot branching | Intent-filtered + Sentence-BERT ranked 50-unit forward window |
| Personalised endings | `Counter`-based play-style profiling → archetype-conditioned generation |
| HuggingFace Transformers | TinyLlama, BART-MNLI, all-MiniLM-L6-v2 |
| PyTorch | Model loading, CUDA/CPU dispatch |
| Gradio | `gr.Blocks` with two-column game/ending layout |
| Plot coherence scores | Per-round Sentence-BERT scores |
| Generation response times | Wall-clock timing per `generate()` call |
| Player choice matching accuracy | BART-MNLI confidence score per round |
| Immersive experience metric | Composite score: coherence×0.5 + intent×0.3 + speed×0.2 |
