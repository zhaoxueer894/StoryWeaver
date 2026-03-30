import gradio as gr
import logging
import os
from core_algorithms import FF7AdventureEngine

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "storyweaver.log")


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Clean up root logger (remove any StreamHandler added by basicConfig, etc.)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Attach the handler DIRECTLY to the "storyweaver" logger so that logs are
    # written to file even if Gradio later modifies the root logger's level or
    # handlers.  propagate=False prevents double-logging via root.
    sw_logger = logging.getLogger("storyweaver")
    for handler in list(sw_logger.handlers):
        sw_logger.removeHandler(handler)
    sw_logger.setLevel(logging.INFO)
    sw_logger.addHandler(file_handler)
    sw_logger.propagate = False


setup_logging()
logger = logging.getLogger("storyweaver")
logger.info(f"Starting StoryWeaver app. Log file: {LOG_FILE}")

# 初始化游戏引擎（仅一次，重置时调用 game.reset() 而非重新构造）
game = FF7AdventureEngine()
logger.info(f"Game initialized. Initial plot unit: {game.current_plot_unit_id}, "
            f"current characters: {game.current_characters}")
logger.info(f"Action options: {game.action_options}")


# ==============================================================================
# 辅助函数
# ==============================================================================

def format_options_display():
    labels = ["A", "B", "C"]
    lines = [f"{labels[i]}. {opt}" for i, opt in enumerate(game.action_options)]
    return "\n".join(lines)


def format_rounds_remaining():
    remaining = game.MAX_ROUNDS - game.round_count
    return f"Remaining interactions: {remaining} / {game.MAX_ROUNDS}"


def format_metrics(metrics: dict) -> str:
    coherence_list = ", ".join(f"{s:.4f}" for s in metrics["coherence_scores"])
    response_list  = ", ".join(f"{t}s" for t in metrics["response_times"])
    choices_list   = "\n".join(f"  Round {i+1}: {c}" for i, c in enumerate(metrics["choices_made"]))

    def rating(score: float) -> str:
        if score >= 0.80: return "Excellent"
        if score >= 0.65: return "Good"
        if score >= 0.50: return "Fair"
        return "Needs Improvement"

    return (
        "=" * 52 + "\n"
        "           PERFORMANCE EVALUATION\n"
        "=" * 52 + "\n\n"
        f"Rounds Played          : {metrics['rounds_played']}\n\n"
        "--- Narrative Quality ---\n"
        f"  Plot Coherence (avg) : {metrics['avg_plot_coherence']:.4f}  [{rating(metrics['avg_plot_coherence'])}]\n"
        f"  Coherence per Round  : [{coherence_list}]\n\n"
        "--- Interaction Responsiveness ---\n"
        f"  Avg Response Time    : {metrics['avg_response_time_sec']:.2f} s\n"
        f"  Response Times       : [{response_list}]\n\n"
        "--- Player Choice Matching ---\n"
        f"  Intent Confidence    : {metrics['avg_intent_confidence']:.4f}  [{rating(metrics['avg_intent_confidence'])}]\n"
        f"  Choices Made         :\n{choices_list}\n\n"
        "--- Immersive Experience ---\n"
        f"  Immersion Score      : {metrics['immersion_score']:.4f}  [{rating(metrics['immersion_score'])}]\n"
        "  (Coherence×0.5 + Intent×0.3 + Speed×0.2)\n\n"
        "=" * 52
    )


# ==============================================================================
# 事件处理
# ==============================================================================

_BACKGROUND_PLACEHOLDER = "Select a role above to begin your adventure — the story background will appear here."


def handle_role_selection(role_name: str):
    if role_name and game.set_player_role(role_name):
        role_status = f"Role selected: {role_name}"
        intro_text = game.introduce_story()   # pinned as permanent story background
    else:
        role_status = "Select a valid role from the dropdown."
        intro_text = _BACKGROUND_PLACEHOLDER

    # Returns: role_status, story_background, options_display, rounds_display
    return role_status, intro_text, format_options_display(), format_rounds_remaining()


def handle_player_choice(choice_idx: int):
    if not game.player_role:
        return (
            gr.update(),           # story_background unchanged
            "Please select your role first.",
            format_options_display(),
            gr.update(),           # game_col unchanged
            gr.update(),           # ending_col unchanged
            "",                    # ending_story_box
            "",                    # metrics_box
            format_rounds_remaining(),
        )

    result = game.player_action(choice_idx)

    story_display = (
        f"{result['full_history']}\n\n"
        f"--- Consistency Score (latest): {result['consistency_score']:.4f} ---"
    )

    if result["game_over"]:
        conclusion = game.conclude_story()
        return (
            gr.update(),                               # story_background unchanged
            conclusion["full_history"],
            format_options_display(),
            gr.update(visible=False),                  # hide game_col
            gr.update(visible=True),                   # show ending_col
            conclusion["ending_text"],
            format_metrics(conclusion["metrics"]),
            format_rounds_remaining(),
        )

    return (
        gr.update(),           # story_background unchanged
        story_display,
        format_options_display(),
        gr.update(),           # game_col unchanged
        gr.update(),           # ending_col unchanged
        "",                    # ending_story_box
        "",                    # metrics_box
        format_rounds_remaining(),
    )


def handle_restart():
    game.reset()
    return (
        _BACKGROUND_PLACEHOLDER,                   # story_background reset to placeholder
        "",                                        # story_out
        format_options_display(),                  # options_display
        gr.update(visible=True),                   # show game_col
        gr.update(visible=False),                  # hide ending_col
        "",                                        # ending_story_box
        "",                                        # metrics_box
        format_rounds_remaining(),                 # rounds_display
        "Please select a role from the dataset.",  # role_status
    )


# ==============================================================================
# Gradio 界面
# ==============================================================================
GAME_OUTPUTS = None   # filled after component definitions

with gr.Blocks(title="StoryWeaver | FF7 Text Adventure") as demo:
    gr.Markdown("""
    # StoryWeaver - Final Fantasy VII Text Adventure
    ## Natural Language Processing | COMP5423 Project
    *Context-Aware Generation | Zero-Shot Intent | Plot Consistency*
    """)

    # ------------------------------------------------------------------
    # 主游戏区域
    # ------------------------------------------------------------------
    with gr.Column(visible=True) as game_col:
        with gr.Row():
            story_background = gr.Textbox(
                label="Story Background",
                lines=10,
                value=_BACKGROUND_PLACEHOLDER,
                interactive=False
            )
        with gr.Row():
            role_selector = gr.Dropdown(
                label="Choose Your Role",
                choices=game.available_roles,
                value=game.available_roles[0] if game.available_roles else None,
                interactive=True
            )
            role_status = gr.Textbox(label="Selected Role", lines=1,
                                     value="Please select a role from the dataset.")
            role_button = gr.Button("Confirm Role", variant="primary")
        with gr.Row():
            story_out = gr.Textbox(label="Story", lines=14)

        gr.Markdown("### Choose Your Action")
        with gr.Row():
            options_display = gr.Textbox(
                label="Available Actions (generated by model)",
                lines=4,
                value=format_options_display(),
                interactive=False
            )
            rounds_display = gr.Textbox(
                label="Progress",
                lines=1,
                value=format_rounds_remaining(),
                interactive=False,
                scale=1
            )
        with gr.Row():
            btn_a = gr.Button("A", variant="primary")
            btn_b = gr.Button("B", variant="secondary")
            btn_c = gr.Button("C", variant="stop")

    # ------------------------------------------------------------------
    # 结局 / 评价区域（初始隐藏）
    # ------------------------------------------------------------------
    with gr.Column(visible=False) as ending_col:
        gr.Markdown("# The End")
        with gr.Row():
            ending_story_box = gr.Textbox(label="Story Ending", lines=12,
                                          interactive=False)
        with gr.Row():
            metrics_box = gr.Textbox(label="Performance Evaluation", lines=22,
                                     interactive=False)
        with gr.Row():
            restart_btn = gr.Button("Restart Game", variant="primary")

    # ------------------------------------------------------------------
    # 公共输出列表（game buttons 和 restart 共用前7项）
    # ------------------------------------------------------------------
    GAME_OUTPUTS = [
        story_background, story_out, options_display,
        game_col, ending_col,
        ending_story_box, metrics_box,
        rounds_display,
    ]

    # 角色选择（story_background 作为固定背景写入一次，不再更新）
    role_button.click(
        handle_role_selection,
        inputs=[role_selector],
        outputs=[role_status, story_background, options_display, rounds_display]
    )

    # 行动按钮
    btn_a.click(lambda: handle_player_choice(0), outputs=GAME_OUTPUTS)
    btn_b.click(lambda: handle_player_choice(1), outputs=GAME_OUTPUTS)
    btn_c.click(lambda: handle_player_choice(2), outputs=GAME_OUTPUTS)

    # 重新开始
    restart_btn.click(
        handle_restart,
        outputs=GAME_OUTPUTS + [role_status]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True, quiet=True)
