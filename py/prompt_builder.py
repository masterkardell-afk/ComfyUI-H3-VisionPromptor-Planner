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
                           custom_system_prompt, n_images,
                           style_directive="", creative_boost="off", motif_hints=None):
    """Build the (system, user) pair for the single planner generation pass.

    The system prompt embeds the full LongMedia Cameras vocabulary
    (camera_knowledge.render_knowledge_block) unless a custom system prompt
    is supplied — in which case it replaces the built-in planner prompt
    entirely (advanced usage).

    Creative boost (fork v1.1.5/v1.1.6):
      - style_directive: the user's own brief (genre / mood / visual style);
        injected verbatim into the user message whenever non-empty —
        including level 'off' (a user's explicit words must always pass).
      - creative_boost: 'standard' appends the CREATIVE WRITING STANDARDS
        block to the built-in system prompt (custom system prompts are left
        untouched); 'expressive' additionally injects the world-spine motif
        hints into the user message and widens the word budget band.
      - 'off' + empty style_directive -> no creative scaffolding injected.
    """
    from . import camera_knowledge
    from . import planner_mode

    level, _level_warning = planner_mode.normalize_creative_boost(creative_boost)

    if (custom_system_prompt or "").strip():
        system = custom_system_prompt.strip()
    else:
        system = load_planner_system_template().replace(
            PLANNER_SYSTEM_PLACEHOLDER, camera_knowledge.render_knowledge_block())
        # Creative standards ride on the BUILT-IN planner prompt only — a
        # user-supplied custom system prompt is advanced usage and stays
        # exactly as written.
        standards = planner_mode.render_creative_standards(level)
        if standards:
            system = system + "\n\n" + standards

    user_parts: list[str] = []
    if vision_context and vision_context.strip():
        user_parts.append(
            "Reference image analysis — RAW MATERIAL for identity anchors ONLY, never "
            "content to copy into the plan: extract solely what the user's idea asks to "
            "take from each picture and never describe the photo itself anywhere in the "
            "answer. <Picture N> refers to the Nth provided image (the user may also "
            "write <image_N-1>: <image_0> == <Picture 1>, <image_1> == <Picture 2>, ...):\n"
            + vision_context.strip())

    idea = (user_idea or "").strip()
    if idea:
        user_parts.append("User idea:\n" + idea)
    else:
        user_parts.append("User idea: (none provided — invent a coherent, concrete cinematic "
                          "sequence consistent with the reference image analysis, if any.)")

    # The user's own creative brief: always passes through when non-empty,
    # at any creative_boost level (level 'off' only disables the fork's own
    # scaffolding, never the author's words).
    directive = (style_directive or "").strip()
    if directive:
        user_parts.append(
            "Creative direction (the author's brief — follow it closely; it may override "
            "default stylistic choices but never the output contract):\n" + directive)

    durations_repr = ", ".join(f"{d:g}s" for d in clip_durations)
    user_parts.append(
        f"Constraint: the final video is {float(total_duration):g} seconds in total, delivered as "
        f"{int(clip_count)} consecutive clips with a target clip length of "
        f"{float(target_clip_duration):g}s. Planned per-clip durations: {durations_repr}. "
        f"Write exactly {int(clip_count)} clips (clip_1 .. clip_{int(clip_count)}) and exactly "
        f"{int(clip_count)} camera cards. The clips form ONE continuous evolving scene: "
        f"clip_1 establishes the situation and every next clip continues the state left "
        f"by the previous clip.")

    # Long clips (beyond the standard ~15s range): spell out per-clip narrative
    # word budgets so the plan's density scales with each clip's duration
    # instead of the contract's default 50-90-word band.
    if clip_durations and max(float(d) for d in clip_durations) > planner_mode.STANDARD_CLIP_SECONDS:
        budgets = ", ".join(
            f"clip_{i + 1} ~{planner_mode.per_clip_word_budget(d, level)} words"
            for i, d in enumerate(clip_durations[:int(clip_count)]))
        user_parts.append(
            "Per-clip word budgets (long clips — scale each clip's narrative to its duration, "
            "covering the full action progression from its start to its end): " + budgets + ".")

    # Expressive level: world-spine + per-clip beat ingredients, sampled by
    # the node (deterministic in the generation seed) from the built-in motif
    # pools. The spine is shared by every clip (LongMedia continuity law:
    # constants live in the world state), each clip's beat is the change it
    # contributes — variety without independent-clip drift.
    if level == "expressive" and motif_hints:
        user_parts.append(
            "Suggested visual motifs — creative ingredients. The WORLD SPINE must stay "
            "present and gradually develop in every clip; each clip's beat is the change "
            "that clip contributes to the ongoing scene. Weave them naturally into the "
            "prose; never quote them verbatim, never list or enumerate them:\n"
            + planner_mode.render_motif_hints_block(motif_hints))

    if int(n_images) > 0:
        user_parts.append(
            f"{int(n_images)} reference picture(s) are attached to this generation; use "
            "<Picture k> labels in the clip text wherever the referenced subject or appearance "
            "must remain visible and consistent. Notation mapping: <image_0> means <Picture 1>, "
            "<image_1> means <Picture 2>, and so on — the user's idea may use either form.")
    else:
        user_parts.append("No reference pictures are attached; do not use <Picture> labels.")

    if extra_instructions and extra_instructions.strip():
        user_parts.append("Additional instructions from the user (respect them unless they "
                          "conflict with the output contract):\n" + extra_instructions.strip())

    user_parts.append("Language: write the plan in English. Dialogue inside <d>...</d> keeps "
                      "its own language verbatim. Remember: no camera language anywhere in the "
                      "GLOBAL or CLIPS sections.")

    return system, "\n\n".join(user_parts)


def build_translate_messages(en_text, preserve_vividness=False):
    """Build the (system, user) pair for the EN -> RU mirror pass.

    preserve_vividness=True (creative_boost on, fork v1.1.5) appends the
    artistic-accuracy note so the translation keeps the vivid sensory
    language instead of flattening it to neutral Russian; False leaves the
    translate system prompt byte-identical to v1.1.4.
    """
    from . import planner_mode

    system = _read_prompt_file("planner_translate.txt")
    if preserve_vividness:
        system = system + "\n\n" + planner_mode.TRANSLATE_VIVID_NOTE
    return system, en_text
