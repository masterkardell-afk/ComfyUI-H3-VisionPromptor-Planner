"""Prompt assembly: task detection, system-prompt loading, word budgets, and
(system, user) message construction for the H3 prompt-writing generation.
"""

from __future__ import annotations

import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

TASK_TYPES = ("T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA")


def detect_task(task_type: str, n_images: int) -> str:
    """Resolve the effective task. Auto: 0 images -> T2VA, 1 -> I2VA, 2 -> FL2VA, >=3 -> Ref2VA."""
    t = (task_type or "Auto").strip()
    if t.lower() != "auto":
        for known in TASK_TYPES:
            if t.lower() == known.lower():
                return known
        raise ValueError(f"[H3 VisionPromptor] Unknown task_type '{task_type}'. "
                         f"Expected one of Auto, {', '.join(TASK_TYPES)}.")
    if n_images <= 0:
        return "T2VA"
    if n_images == 1:
        return "I2VA"
    if n_images == 2:
        return "FL2VA"
    return "Ref2VA"


def _read_prompt_file(*parts: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, *parts), "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt(task: str) -> str:
    """Base H3 system prompt + task addendum, re-read from disk on every run."""
    base = _read_prompt_file("h3_system_base.txt")
    addendum = _read_prompt_file("tasks", f"{task.lower()}.txt")
    return base + "\n\n" + addendum


def word_budget(task: str, duration: float) -> int:
    """Target body length in words. Base modes: duration*25. Ref2VA's
    detailed_description targets 350-500 words -> duration*35 capped at 500."""
    if task == "Ref2VA":
        return min(int(duration * 35), 500)
    return int(duration * 25)


def build_messages(task, duration, user_idea, vision_context, extra_instructions,
                   output_language, custom_system_prompt) -> tuple[str, str]:
    """Build the (system, user) pair for the prompt-writing generation."""
    system = (custom_system_prompt or "").strip() or load_system_prompt(task)

    budget = word_budget(task, duration)
    frames = int(duration * 24)

    user_parts: list[str] = []
    if vision_context and vision_context.strip():
        user_parts.append(
            "Reference image analysis (facts about the provided picture(s); "
            "<Picture N> refers to the Nth provided image):\n" + vision_context.strip()
        )

    idea = (user_idea or "").strip()
    if idea:
        user_parts.append("User idea:\n" + idea)
    else:
        user_parts.append("User idea: (none provided — invent a coherent, concrete scene "
                          "consistent with the reference image analysis, if any.)")

    user_parts.append(f"Constraint: The video will be {duration} seconds long (approx. {frames} frames).")

    if task == "Ref2VA":
        user_parts.append(f"Write the detailed_description section at roughly {budget} words "
                          "(target range 350-500 words; prioritize a complete spoken timeline "
                          "over mechanically hitting the count).")
    else:
        user_parts.append(f"Write the integrated_multimodal_description body at roughly {budget} words.")

    if extra_instructions and extra_instructions.strip():
        user_parts.append("Additional instructions from the user (respect them unless they "
                          "conflict with the H3 format rules):\n" + extra_instructions.strip())

    if (output_language or "English").strip().lower().startswith("chinese"):
        user_parts.append("Language: write the prompt body in Simplified Chinese (简体中文), "
                          "but keep all field names, shot labels, reference labels, speaker IDs, "
                          "and dialogue/lyrics in their original language exactly as provided.")
    else:
        user_parts.append("Language: write the prompt body in English. Dialogue, lyrics, and "
                          "on-screen text keep their original language verbatim.")

    return system, "\n\n".join(user_parts)


# ---------------------------------------------------------------------------
# LongMedia Planner mode (fork)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PLACEHOLDER = "{{CAMERA_KNOWLEDGE_BLOCK}}"


def load_planner_system_template() -> str:
    """Raw planner system prompt with the {{CAMERA_KNOWLEDGE_BLOCK}} placeholder."""
    return _read_prompt_file("planner_system.txt")


def build_planner_messages(total_duration, target_clip_duration, clip_count, clip_durations,
                           user_idea, vision_context, extra_instructions,
                           custom_system_prompt, n_images):
    """Build the (system, user) pair for the single planner generation pass.

    The system prompt embeds the full LongMedia Cameras vocabulary
    (camera_knowledge.render_knowledge_block) unless a custom system prompt
    is supplied — in which case it replaces the built-in planner prompt
    entirely (advanced usage).
    """
    from . import camera_knowledge

    if (custom_system_prompt or "").strip():
        system = custom_system_prompt.strip()
    else:
        system = load_planner_system_template().replace(
            PLANNER_SYSTEM_PLACEHOLDER, camera_knowledge.render_knowledge_block())

    user_parts: list[str] = []
    if vision_context and vision_context.strip():
        user_parts.append(
            "Reference image analysis (facts about the provided picture(s); "
            "<Picture N> refers to the Nth provided image):\n" + vision_context.strip())

    idea = (user_idea or "").strip()
    if idea:
        user_parts.append("User idea:\n" + idea)
    else:
        user_parts.append("User idea: (none provided — invent a coherent, concrete cinematic "
                          "sequence consistent with the reference image analysis, if any.)")

    durations_repr = ", ".join(f"{d:g}s" for d in clip_durations)
    user_parts.append(
        f"Constraint: the final video is {float(total_duration):g} seconds in total, delivered as "
        f"{int(clip_count)} consecutive clips with a target clip length of "
        f"{float(target_clip_duration):g}s. Planned per-clip durations: {durations_repr}. "
        f"Write exactly {int(clip_count)} clips (clip_1 .. clip_{int(clip_count)}) and exactly "
        f"{int(clip_count)} camera cards.")

    if int(n_images) > 0:
        user_parts.append(
            f"{int(n_images)} reference picture(s) are attached to this generation; use "
            "<Picture k> labels in the clip text wherever the referenced subject or appearance "
            "must remain visible and consistent.")
    else:
        user_parts.append("No reference pictures are attached; do not use <Picture> labels.")

    if extra_instructions and extra_instructions.strip():
        user_parts.append("Additional instructions from the user (respect them unless they "
                          "conflict with the output contract):\n" + extra_instructions.strip())

    user_parts.append("Language: write the plan in English. Dialogue inside <d>...</d> keeps "
                      "its own language verbatim. Remember: no camera language anywhere in the "
                      "GLOBAL or CLIPS sections.")

    return system, "\n\n".join(user_parts)


def build_translate_messages(en_text):
    """Build the (system, user) pair for the EN -> RU mirror pass."""
    system = _read_prompt_file("planner_translate.txt")
    return system, en_text
