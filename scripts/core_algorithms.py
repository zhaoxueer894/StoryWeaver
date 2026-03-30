import logging
import os
import re
import heapq
import time
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

# HuggingFace 大模型库
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging as transformers_logging
)
# 语义相似度 / 一致性检测
from sentence_transformers import SentenceTransformer, util

# 项目统一日志器（handler 由 04_app.py 的 setup_logging 统一配置）
logger = logging.getLogger("storyweaver")

# 关闭 Transformers 冗余日志
transformers_logging.set_verbosity_error()

# ==============================================================================
# 全局路径配置（严格加载你预处理的 4 个数据集）
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 加载全部 processed 数据（核心：4个文件全部深度使用）
FF7_CLEANED_BASE = pd.read_csv(os.path.join(PROCESSED_DIR, "ff7_cleaned_base.csv"))
FF7_DIALOGUE_CORPUS = pd.read_csv(os.path.join(PROCESSED_DIR, "dialogue_dataset.csv"))
FF7_PLOT_UNITS = pd.read_csv(os.path.join(PROCESSED_DIR, "plot_units.csv"))
FF7_CONSISTENCY_ANNOTATIONS = pd.read_csv(os.path.join(PROCESSED_DIR, "plot_consistency_samples.csv"))

character_roles_path = os.path.join(PROCESSED_DIR, "character_roles.csv")
if os.path.exists(character_roles_path):
    FF7_CHARACTER_ROLES = pd.read_csv(character_roles_path)
else:
    chars = []
    for _, row in FF7_PLOT_UNITS.iterrows():
        for char in [c.strip() for c in str(row["character_list"]).split(",") if c.strip()]:
            chars.append(char)
    role_df = pd.DataFrame({"character": chars})
    FF7_CHARACTER_ROLES = role_df["character"].value_counts().reset_index()
    FF7_CHARACTER_ROLES.columns = ["character", "plot_unit_count"]
    FF7_CHARACTER_ROLES["plot_unit_ids"] = ""

# 设备自动配置（CPU/GPU 通用）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 模块 1：优化版 TinyLlama 剧情生成（100% 不胡言乱语）
# ==============================================================================
class FF7ContextualGenerator:
    def __init__(self):
        # 稳定轻量模型，专为英文剧情生成优化
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # 修复CPU加载，添加信任代码
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(DEVICE)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = 896
        self.max_new_tokens = 180        # increased: 120 was too short and caused mid-sentence cut-offs
        self.temperature = 0.5
        self.top_p = 0.9
        self.repetition_penalty = 1.25
        self.last_prompt = ""
        self.last_generated_text = ""

    @staticmethod
    def _trim_to_sentence(text: str) -> str:
        """Clip text at the last sentence-ending punctuation (.  !  ?) so the
        output never ends mid-sentence regardless of token budget."""
        last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        return text[:last_end + 1].strip() if last_end > 0 else text.strip()

    @staticmethod
    def _normalize_option(text: str) -> str:
        """Rewrite a raw LLM option into second-person 'You …' form.

        Options describe only the player's own actions.  Any character-name
        prefix ('As Cloud, you …', 'As Barret …') is stripped so that options
        never refer to another character as the acting subject.
        Returns an empty string if the text is invalid or un-salvageable.
        """
        # Strip leading label artifacts: A.  B.  C.  1.  2)  etc.
        text = re.sub(r'^[A-Ca-c1-3][\.\)]\s*', '', text).strip().strip('"\'')

        # ── Hard rejection: un-salvageable content ─────────────────────────────
        # 1. Template placeholder leak: model echoed "[action]" or "[...]" literally.
        if '[' in text or ']' in text:
            return ""
        # 2. NPC dialogue attribution right after a "You" prefix, e.g. "You (Barret) said…"
        if re.search(r'(?i)^you\s*\(', text):
            return ""
        # 3. Pure meta-commentary.
        if re.search(r'(?i)(this line will|always appear|format:|example:|note:)', text):
            return ""

        # ── Strip any "As <name>, you" or "As <name>" prefix ──────────────────
        # This removes both the player's own name and any NPC name the model
        # may have accidentally placed here.
        text = re.sub(r'(?i)^as [^,]+,\s*you\b\s*', '', text).strip()
        text = re.sub(r'(?i)^as [^,\s]+[,\s]+', '', text).strip()

        # ── Normalise to start with capitalised "You" ──────────────────────────
        if text.lower().startswith('you '):
            text = 'You ' + text[4:]
        else:
            # Prepend "You" and lower-case the first original character.
            text = 'You ' + text[:1].lower() + text[1:]

        # Guarantee sentence-ending punctuation.
        if text and text[-1] not in '.!?':
            text += '.'
        # Reject fragments shorter than 5 words.
        return text if len(text.split()) >= 5 else ""

    def build_chat_prompt(
        self,
        current_scene: str,
        player_choice: str,
        characters: str,
        memory_prompt: str,
        story_history: str = ""
    ) -> str:
        """
        🔥 核心修复：严格使用 TinyLlama 官方 Chat 模板
        强制模型只生成 FF7 剧情，无任何废话/技术内容
        """
        # Truncate history to last 512 chars to stay within token budget
        recent_history = story_history[-512:].strip() if story_history else ""
        messages = [
            {
                "role": "system",
                "content": f"You are a Final Fantasy VII writer. Only write short, canonical FF7 scene narrative (2-3 sentences). No meta text, no code, no instructions. Characters: {characters}"
            },
            {
                "role": "user",
                "content": f"Memory: {memory_prompt}\nStory so far: {recent_history}\nCurrent Scene: {current_scene}\nPlayer Action: {player_choice}\nContinue the FF7 story:"
            }
        ]
        # 官方模板，保证输出纯净
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        self.last_prompt = prompt
        return prompt

    def generate(
        self,
        story_history: str,
        current_scene: str,
        player_choice: str,
        characters: str,
        memory_prompt: str = None,
        reference_scene: str = None,
        reference_plot_unit_id: int = None
    ) -> str:
        prompt = self.build_chat_prompt(
            current_scene, player_choice, characters, memory_prompt or "No memory",
            story_history=story_history or ""
        )
        logger.info(f"[GENERATION PROMPT]\n{prompt}")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        ).to(DEVICE)

        # 生成输出
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 纯剧情提取，过滤所有指令；裁剪到最后一个完整句子
        output_text = self._trim_to_sentence(self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ))

        logger.info(f"[GENERATION RESPONSE]\n{output_text}")
        self.last_generated_text = output_text
        return output_text if len(output_text) > 5 else reference_scene

    def generate_intro(self, scene: str, characters: str, player_role: str) -> str:
        """Generate an opening narrative that sets the story background before the first player action."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Final Fantasy VII narrator. "
                    "Write a vivid 3-4 sentence opening passage that introduces the story setting, "
                    "the atmosphere, and the player's character. "
                    "Be immersive and engaging. No meta text, no instructions, only narrative prose."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Player character: {player_role}\n"
                    f"Characters present: {characters}\n"
                    f"Opening scene: {scene}\n"
                    "Begin the story:"
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info(f"[INTRO PROMPT]\n{prompt}")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length
        ).to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        intro = self._trim_to_sentence(self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ))
        logger.info(f"[INTRO RESPONSE]\n{intro}")
        return intro if len(intro) > 10 else scene

    def generate_ending(
        self,
        story_history: str,
        characters: str,
        player_role: str,
        choices_summary: str = "",
        memory_prompt: str = ""
    ) -> str:
        """Generate a conclusive ending narrative for the full adventure.

        The ending adapts to the player's demonstrated play style (choices_summary)
        and the distilled key memories of the journey (memory_prompt), so different
        players receive genuinely different conclusions.
        """
        recent_history = story_history[-768:].strip() if story_history else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Final Fantasy VII narrator writing the story's conclusion. "
                    "Write a satisfying 4-5 sentence ending that reflects the specific journey "
                    f"{player_role} took — their choices, their style, and the events they witnessed. "
                    "The ending MUST differ meaningfully based on whether the player was a diplomat, "
                    "an explorer, or a warrior. No meta text, no instructions, only narrative prose."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Player character: {player_role}\n"
                    f"Play style across all rounds: {choices_summary}\n"
                    f"Key memories from the journey: {memory_prompt}\n"
                    f"Characters present at the end: {characters}\n"
                    f"Final moments of the story: {recent_history}\n"
                    "Write the ending:"
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info(f"[ENDING PROMPT]\n{prompt}")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length
        ).to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        ending = self._trim_to_sentence(self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ))
        logger.info(f"[ENDING RESPONSE]\n{ending}")
        return ending if len(ending) > 10 else "The adventure comes to a close..."

    def generate_options(
        self,
        current_scene: str,
        characters: str,
        story_history: str = "",
        player_role: str = "you"
    ) -> list:
        """Use the LLM to generate 3 distinct, character-perspective action choices (A/B/C).

        If the first attempt contains duplicates, empty items, or wrong format, a second
        attempt is made with a correction prompt describing the specific problem.  Only if
        both attempts fail does the method return [] to trigger the static fallback.
        """
        recent_history = story_history[-256:].strip() if story_history else ""

        def _build_messages(correction_note: str = "") -> list:
            system_content = (
                f"You are a Final Fantasy VII game master. The player controls {player_role}. "
                f"Generate exactly 3 DISTINCT action choices the player can take. "
                f"Rules:\n"
                f"1. Every sentence MUST start with 'You' (second person, the player's own action).\n"
                f"2. Every sentence must be a complete action (8-15 words).\n"
                f"3. Each choice must describe a DIFFERENT category of action: "
                f"one social/talk, one explore/observe, one move/combat.\n"
                f"4. No two choices may be paraphrases of each other.\n"
                f"5. Do NOT write character names as the subject. Do NOT write NPC dialogue, "
                f"character names in parentheses, or any meta-commentary.\n"
                f"Output ONLY these three lines — no other text:\n"
                f"A. You speak with the others about the current situation.\n"
                f"B. You move forward carefully to scout the area ahead.\n"
                f"C. You examine the surroundings for anything unusual.\n"
                f"Replace those example actions with 3 NEW actions that fit the current scene."
            )
            if correction_note:
                system_content += f"\n\nCORRECTION REQUIRED: {correction_note}"
            return [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"Characters present: {characters}\n"
                        f"Recent story: {recent_history}\n"
                        f"Current scene: {current_scene}\n"
                        f"Generate 3 distinct choices for {player_role}:"
                    )
                }
            ]

        def _collect(candidates: list) -> list:
            """Normalize each candidate and deduplicate case-insensitively."""
            seen, out = set(), []
            for item in candidates:
                cleaned = self._normalize_option(item)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(cleaned)
            return out

        def _attempt(messages: list, attempt_num: int) -> list:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.info(f"[OPTIONS PROMPT (attempt {attempt_num})]\n{prompt}")

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(DEVICE)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            raw = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
            ).strip()
            logger.info(f"[OPTIONS RESPONSE (attempt {attempt_num})]\n{raw}")

            # Try labelled matches first (A. … B. … C. …)
            matches = re.findall(r'[ABC]\.\s*(.+)', raw)
            normalized = _collect(matches)
            if len(normalized) >= 3:
                return normalized[:3]
            # Fallback: every non-empty line
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            normalized = _collect(lines)
            if len(normalized) >= 3:
                return normalized[:3]
            return normalized  # partial or empty

        # ── Attempt 1 ──────────────────────────────────────────────────────────
        result = _attempt(_build_messages(), attempt_num=1)
        if len(result) >= 3:
            return result[:3]

        # ── Diagnose and build a targeted correction note ──────────────────────
        if len(result) == 0:
            correction_note = (
                "Your previous response contained NO valid options. "
                "Do NOT write character names as the subject, NPC dialogue, "
                "parenthesised names, or brackets like [action]. "
                "Output ONLY three lines A., B., C. where each line starts "
                "with 'You' followed by a concrete action verb."
            )
        else:
            correction_note = (
                f"Your previous response only produced {len(result)} valid option(s). "
                "You need exactly 3. Each must start with 'You' and describe a "
                "DIFFERENT action. Do not repeat or paraphrase."
            )
        logger.info(
            f"[OPTIONS RETRY] attempt 1 produced {len(result)} valid option(s). "
            f"Retrying with correction: {correction_note}"
        )

        # ── Attempt 2 (with correction) ────────────────────────────────────────
        result = _attempt(_build_messages(correction_note), attempt_num=2)
        if len(result) >= 3:
            return result[:3]
        if result:
            logger.warning(f"[OPTIONS] attempt 2 produced only {len(result)} option(s); using static fallback")
        return []  # signals caller to use static fallback

# ==============================================================================
# 模块 2：零样本意图识别（BART-large-MNLI）
# ==============================================================================
class FF7IntentRecognizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-mnli"
        self.device = DEVICE
        self.classifier = pipeline(
            task="zero-shot-classification",
            model=self.model_name,
            device=self.device
        )
        self.INTENT_LABELS = ["talk", "explore", "move", "interact", "observe"]

    def recognize_intent(self, choice_text: str) -> dict:
        result = self.classifier(choice_text, self.INTENT_LABELS)
        return {
            "intent": result["labels"][0],
            "confidence_score": round(float(result["scores"][0]), 3),
            "all_intents": dict(zip(result["labels"], result["scores"]))
        }

# ==============================================================================
# 模块 3：剧情一致性检测（Sentence-BERT）
# ==============================================================================
class FF7ConsistencyChecker:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.device = DEVICE
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.reference_texts = FF7_CONSISTENCY_ANNOTATIONS["plot_text"].dropna().astype(str).tolist()[:60]
        self.reference_samples_count = len(self.reference_texts)

    def compute_consistency(self, history: str, new_text: str) -> float:
        hist_emb = self.model.encode(history, convert_to_tensor=True)
        new_emb = self.model.encode(new_text, convert_to_tensor=True)
        history_score = util.cos_sim(hist_emb, new_emb).item()

        sample_scores = []
        for ref in self.reference_texts[:20]:
            ref_emb = self.model.encode(ref, convert_to_tensor=True)
            sample_scores.append(util.cos_sim(new_emb, ref_emb).item())
        
        avg_sample_score = sum(sample_scores)/len(sample_scores) if sample_scores else 0.5
        final_score = (history_score * 0.7 + avg_sample_score * 0.3)
        return round(final_score, 4)

# ==============================================================================
# 模块 4：游戏核心引擎（完整保留所有功能）
# ==============================================================================
class FF7AdventureEngine:
    def __init__(self):
        self.generator = FF7ContextualGenerator()
        self.intent_recognizer = FF7IntentRecognizer()
        self.consistency_checker = FF7ConsistencyChecker()
        self.logger = logging.getLogger("storyweaver")
        self.player_role = None
        self.available_roles = self._get_available_roles()
        self.memory_heap = []
        self.memory_counter = 0
        self.memory_size = 6

        self.global_story_history = "[START OF FINAL FANTASY VII ADVENTURE | Midgar Sector 1]\n"
        self.current_plot_unit_id = 1
        self.max_history_length = 3072
        self.MAX_ROUNDS = 10

        # Per-session metrics tracking
        self.round_count = 0
        self.consistency_scores = []
        self.response_times = []
        self.intent_confidences = []
        self.intent_labels = []      # stores the label string per round, e.g. "talk"
        self.choices_made = []

        self.load_plot_unit()
        self.build_action_options()
        self.log_startup_info()

    def _trim_history(self):
        if len(self.global_story_history) > self.max_history_length:
            tail = self.global_story_history[-self.max_history_length:]
            # Advance to the first newline so we never cut mid-sentence.
            newline_pos = tail.find('\n')
            self.global_story_history = tail[newline_pos + 1:] if newline_pos != -1 else tail

    def _parse_characters(self, character_list: str):
        return [c.strip() for c in str(character_list).split(",") if c.strip()]

    def _score_memory(self, text: str) -> float:
        keywords = ["cloud", "sephiroth", "tifa", "aerith", "barret", "shinra", "avalanche", "midgar"]
        score = len(text) / 120.0
        score += sum(text.lower().count(word) * 1.0 for word in keywords)
        return score

    def add_memory_entry(self, text: str):
        trimmed = re.sub(r"\s+", " ", text).strip()
        summary = trimmed if len(trimmed) <= 200 else trimmed[:200].rsplit(" ", 1)[0] + "..."
        score = self._score_memory(trimmed)
        # Use (score, ...) so heappop removes the lowest-scoring entry, keeping the best ones.
        heapq.heappush(self.memory_heap, (score, self.memory_counter, summary))
        self.memory_counter += 1
        if len(self.memory_heap) > self.memory_size:
            heapq.heappop(self.memory_heap)

    def format_memory_prompt(self) -> str:
        if not self.memory_heap:
            return "No memory items available."
        memory_items = sorted(self.memory_heap, reverse=True)  # highest score first
        lines = ["Important memory items:"]
        for idx, item in enumerate(memory_items, start=1):
            lines.append(f"{idx}. {item[2]}")
        return "\n".join(lines)

    def _get_available_roles(self, top_n: int = 12):
        if "character" in FF7_CHARACTER_ROLES.columns and "plot_unit_count" in FF7_CHARACTER_ROLES.columns:
            sorted_roles = FF7_CHARACTER_ROLES.sort_values("plot_unit_count", ascending=False)
            return sorted_roles["character"].head(top_n).tolist()
        elif "character" in FF7_CHARACTER_ROLES.columns:
            return FF7_CHARACTER_ROLES["character"].head(top_n).tolist()
        return []

    def set_player_role(self, role_name: str) -> bool:
        if not role_name:
            return False
        if self.available_roles and role_name not in self.available_roles:
            self.logger.warning(f"Selected role '{role_name}' is not in available roles.")
            return False
        self.player_role = role_name
        self.logger.info(f"Player selected role: {role_name}")
        self.global_story_history += f"\n[PLAYER ROLE: {role_name}]\n"
        return True

    def choose_next_plot_unit(self, player_choice: str, intent: str = None) -> int:
        candidates = FF7_PLOT_UNITS.copy()
        if self.player_role:
            mask = candidates["character_list"].str.contains(self.player_role, case=False, na=False)
            if mask.any():
                candidates = candidates[mask]
        if candidates.empty:
            candidates = FF7_PLOT_UNITS.copy()

        # Use the NLP-recognized intent when available, fall back to string matching.
        effective_intent = (intent or "").lower()
        if effective_intent == "talk":
            priority = ["dialogue", "narration", "action"]
        elif effective_intent in ("move",):
            priority = ["action", "narration", "dialogue"]
        elif effective_intent in ("explore", "observe"):
            priority = ["action", "narration", "dialogue"]
        elif effective_intent == "interact":
            priority = ["action", "dialogue", "narration"]
        else:
            # Fallback: crude string match on the raw choice text
            if "Talk" in player_choice:
                priority = ["dialogue", "narration", "action"]
            elif "Move" in player_choice:
                priority = ["action", "narration", "dialogue"]
            elif "Explore" in player_choice:
                priority = ["action", "narration", "dialogue"]
            else:
                priority = ["dialogue", "action", "narration"]

        selected = None
        for narrative_type in priority:
            subset = candidates[candidates["narrative_types"].str.contains(narrative_type, case=False, na=False)]
            if not subset.empty:
                selected = subset
                break
        if selected is None or selected.empty:
            selected = candidates

        selected = selected[selected["plot_unit_id"] != self.current_plot_unit_id]
        if selected.empty:
            selected = FF7_PLOT_UNITS[FF7_PLOT_UNITS["plot_unit_id"] != self.current_plot_unit_id]

        # Prefer units ahead of the current position to maintain forward narrative momentum.
        ahead = selected[selected["plot_unit_id"] > self.current_plot_unit_id]
        pool = ahead if not ahead.empty else selected

        # Limit the candidate pool to a forward window of 50 units so the story
        # progresses without jumping too far, while still giving the similarity
        # model meaningful variety to choose from.
        pool = pool.head(50)

        if len(pool) == 1:
            return int(pool.iloc[0]["plot_unit_id"])

        # -----------------------------------------------------------------------
        # NLP: use the already-loaded sentence-BERT model (all-MiniLM-L6-v2) to
        # rank candidates by semantic relevance to the player's choice and the
        # recent story context.  Different choices produce different query vectors,
        # so the same pool of candidates produces different next-unit selections.
        # -----------------------------------------------------------------------
        recent_ctx = self.global_story_history[-300:].strip()
        query = f"{player_choice}. {recent_ctx}"
        encoder = self.consistency_checker.model          # reuse loaded model, no extra cost

        query_emb      = encoder.encode(query, convert_to_tensor=True, show_progress_bar=False)
        candidate_texts = pool["full_plot_text"].tolist()
        candidate_ids   = pool["plot_unit_id"].tolist()
        candidate_embs  = encoder.encode(candidate_texts, convert_to_tensor=True, show_progress_bar=False)

        scores   = util.cos_sim(query_emb, candidate_embs)[0]   # shape: (pool_size,)
        best_idx = int(scores.argmax())
        logger.info(
            f"[NAVIGATION] intent={player_choice!r} → selected unit {candidate_ids[best_idx]} "
            f"(sim={scores[best_idx]:.4f}) from pool of {len(pool)}"
        )
        return int(candidate_ids[best_idx])

    def introduce_story(self) -> str:
        """Generate an opening narrative after the player selects a role.

        The intro text is appended to the global story history for continuity
        and also returned on its own so the UI can pin it as the story background.
        """
        intro = self.generator.generate_intro(
            self.current_plot_text,
            self.current_characters,
            self.player_role
        )
        self.global_story_history += f"\n{intro}"
        self._trim_history()
        return intro  # return only the intro, not the full history

    def log_startup_info(self):
        self.logger.info("StoryWeaver engine initialized.")

    def load_plot_unit(self):
        try:
            unit_data = FF7_PLOT_UNITS[FF7_PLOT_UNITS["plot_unit_id"] == self.current_plot_unit_id].iloc[0]
            self.current_plot_text = unit_data["full_plot_text"]
            self.current_characters = unit_data["character_list"]
            self.dialogue_count = unit_data["dialogue_count"]
            self.narrative_types = FF7_CLEANED_BASE[FF7_CLEANED_BASE["plot_unit_id"] == self.current_plot_unit_id]["narrative_type"].value_counts().to_dict()
        except Exception:
            self.current_plot_unit_id = 1
            self.load_plot_unit()

    def build_action_options(self):
        role = self.player_role or "you"
        try:
            options = self.generator.generate_options(
                self.current_plot_text,
                self.current_characters,
                story_history=self.global_story_history,
                player_role=role
            )
            if len(options) == 3:
                self.action_options = options
                return
        except Exception as e:
            self.logger.warning(f"Option generation failed, using static fallback: {e}")

        # Static fallback — second-person sentences, no duplicates.
        candidates = []
        char_list = self.current_characters.lower()
        if any(c in char_list for c in ["barret", "cloud", "jessie", "biggs"]):
            candidates.append("You approach your teammates and discuss the situation ahead.")
        if "action" in self.narrative_types or "move" in self.current_plot_text.lower():
            candidates.append("You move forward cautiously toward the next area.")
        if "explore" in self.current_plot_text.lower() or "system" not in self.narrative_types:
            candidates.append("You scan the surroundings carefully for anything unusual.")
        # Fill remaining slots with distinct fallbacks (never repeat)
        extras = [
            "You stay alert and observe everything happening around you.",
            "You take a moment to gather your thoughts before acting.",
        ]
        for extra in extras:
            if len(candidates) >= 3:
                break
            if extra not in candidates:
                candidates.append(extra)
        self.action_options = candidates[:3]

    def next_plot_unit(self, player_choice: str = None):
        if player_choice:
            self.current_plot_unit_id = self.choose_next_plot_unit(player_choice)
        else:
            self.current_plot_unit_id += 1
        if self.current_plot_unit_id > len(FF7_PLOT_UNITS):
            self.current_plot_unit_id = 1
        self.load_plot_unit()
        self.build_action_options()

    def player_action(self, choice_index: int) -> dict:
        player_choice = self.action_options[choice_index]
        intent_result = self.intent_recognizer.recognize_intent(player_choice)
        self.intent_confidences.append(intent_result["confidence_score"])
        self.intent_labels.append(intent_result["intent"])
        self.choices_made.append(player_choice)

        next_plot_id = self.choose_next_plot_unit(player_choice, intent=intent_result["intent"])
        next_plot_data = FF7_PLOT_UNITS[FF7_PLOT_UNITS["plot_unit_id"] == next_plot_id].iloc[0]
        next_plot_text = next_plot_data["full_plot_text"]

        memory_prompt = self.format_memory_prompt()
        t_start = time.time()
        new_story = self.generator.generate(
            self.global_story_history,
            next_plot_text,
            player_choice,
            next_plot_data["character_list"],
            memory_prompt=memory_prompt,
            reference_scene=next_plot_text
        )
        self.response_times.append(round(time.time() - t_start, 2))

        consistency_score = self.consistency_checker.compute_consistency(self.global_story_history, new_story)
        self.consistency_scores.append(consistency_score)
        self.global_story_history += f"\n\n[PLAYER CHOICE: {player_choice}]\n{new_story}"
        self._trim_history()

        self.add_memory_entry(self.current_plot_text)
        self.add_memory_entry(new_story)
        self.current_plot_unit_id = next_plot_id
        self.round_count += 1
        game_over = self.round_count >= self.MAX_ROUNDS
        self.load_plot_unit()
        if not game_over:
            self.build_action_options()

        return {
            "plot_unit_id": self.current_plot_unit_id,
            "player_choice": player_choice,
            "intent": intent_result,
            "generated_story": new_story,
            "consistency_score": consistency_score,
            "current_characters": self.current_characters,
            "full_history": self.global_story_history,
            "game_over": game_over,
        }

    def conclude_story(self) -> dict:
        """Generate the ending narrative and compute all performance metrics."""
        # ------------------------------------------------------------------
        # NLP: build a play-style profile from the per-round intent labels.
        # The Counter gives the distribution of choices the player made; the
        # dominant intent maps to a narrative archetype that conditions the
        # ending prompt, producing a different ending for a diplomat vs. a
        # warrior vs. an explorer.
        # ------------------------------------------------------------------
        from collections import Counter
        intent_counts = Counter(self.intent_labels)
        total_rounds  = len(self.intent_labels) or 1
        style_map = {
            "talk":     "a diplomat who prioritised conversation and alliance-building",
            "explore":  "an explorer who sought hidden paths and uncovered secrets",
            "move":     "a decisive warrior who pressed forward without hesitation",
            "interact": "a problem-solver who engaged directly with every obstacle",
            "observe":  "a careful strategist who studied situations before acting",
        }
        dominant_intent = intent_counts.most_common(1)[0][0] if intent_counts else "explore"
        play_style_desc = style_map.get(dominant_intent, "a versatile hero")
        distribution    = ", ".join(
            f"{lbl}: {cnt}/{total_rounds}" for lbl, cnt in intent_counts.most_common()
        )
        recent_choices  = "; ".join(self.choices_made[-5:])
        choices_summary = (
            f"{play_style_desc}. "
            f"Intent distribution — {distribution}. "
            f"Last choices: {recent_choices}."
        )

        memory_prompt = self.format_memory_prompt()
        logger.info(f"[ENDING] play style: {choices_summary}")

        ending_text = self.generator.generate_ending(
            self.global_story_history,
            self.current_characters,
            self.player_role or "the hero",
            choices_summary=choices_summary,
            memory_prompt=memory_prompt
        )
        self.global_story_history += f"\n\n[THE END]\n{ending_text}"

        avg_coherence = round(sum(self.consistency_scores) / len(self.consistency_scores), 4) if self.consistency_scores else 0.0
        avg_response_time = round(sum(self.response_times) / len(self.response_times), 2) if self.response_times else 0.0
        avg_intent_conf = round(sum(self.intent_confidences) / len(self.intent_confidences), 4) if self.intent_confidences else 0.0
        # Immersive experience: weighted composite (coherence 50%, intent accuracy 30%, speed bonus 20%)
        speed_bonus = round(min(1.0, 1.0 / avg_response_time) if avg_response_time > 0 else 0.0, 4)
        immersion_score = round(avg_coherence * 0.5 + avg_intent_conf * 0.3 + speed_bonus * 0.2, 4)

        self.logger.info(
            f"[GAME OVER] rounds={self.round_count}, avg_coherence={avg_coherence}, "
            f"avg_response_time={avg_response_time}s, avg_intent_conf={avg_intent_conf}, "
            f"immersion={immersion_score}"
        )
        return {
            "ending_text": ending_text,
            "full_history": self.global_story_history,
            "metrics": {
                "rounds_played": self.round_count,
                "avg_plot_coherence": avg_coherence,
                "coherence_scores": self.consistency_scores,
                "avg_response_time_sec": avg_response_time,
                "response_times": self.response_times,
                "avg_intent_confidence": avg_intent_conf,
                "immersion_score": immersion_score,
                "choices_made": self.choices_made,
            },
        }

    def reset(self):
        """Reset game state for a new session without reloading the ML models."""
        self.player_role = None
        self.memory_heap = []
        self.memory_counter = 0
        self.global_story_history = "[START OF FINAL FANTASY VII ADVENTURE | Midgar Sector 1]\n"
        self.current_plot_unit_id = 1
        self.round_count = 0
        self.consistency_scores = []
        self.response_times = []
        self.intent_confidences = []
        self.intent_labels = []
        self.choices_made = []
        self.load_plot_unit()
        self.build_action_options()
        self.logger.info("Game reset for new session.")