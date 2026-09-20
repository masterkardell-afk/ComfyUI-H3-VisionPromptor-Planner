"""ComfyUI V3 node definitions for H3 Vision Promptor.

Mirrors the native TextGenerate node (comfy_extras/nodes_textgen.py) for the
tokenize/generate/decode call pattern, and the proven io.Autogrow usage with
io.Autogrow.TemplatePrefix for multi-image inputs.

Top-level comfy imports (folder_paths, comfy_api.latest) are acceptable here —
this module is only imported inside ComfyUI (or by tests that stub them).
"""

from __future__ import annotations

import json
import time

import folder_paths
from comfy_api.latest import io

from . import engine, vision, prompt_builder, post_processor, planner_mode, camera_knowledge

CATEGORY = "H3/Promptor"
NONE_OPTION = "<none - use CLIP input>"
SKIP_VISION = "skip (idea only)"
CUSTOM_VISION = "custom"
TASK_OPTIONS = ["Auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]
MAX_SEED = 0xFFFFFFFF

# Planner-mode constants (fork).
PLANNER_OUTPUT_NAMES = (
    "planner_prompt", "planner_prompt_ru", "global_prompt",
    "clip_durations", "camera_params", "reference_images_status",
)
TRANSLATE_TEMPERATURE = 0.3


def _text_encoder_options() -> list[str]:
    try:
        return [NONE_OPTION] + folder_paths.get_filename_list("text_encoders")
    except Exception:
        return [NONE_OPTION]


def _vision_preset_keys() -> list[str]:
    return list(vision.load_vision_presets().keys())


def _images_autogrow(min_images: int) -> io.Autogrow.Input:
    """Multi-image input as a growable group of IMAGE sockets (image_0..image_3),
    following the Autogrow.TemplatePrefix pattern used in the wild."""
    return io.Autogrow.Input(
        "images",
        optional=True,
        tooltip="Reference image(s). Add up to 4. 1 image = I2VA/Ref2VA, 2 = FL2VA, 3+ = Ref2VA.",
        template=io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image", tooltip="Reference image"),
            prefix="image_",
            min=min_images,
            max=4,
        ),
    )


def _collect_images(images) -> "object | None":
    """Normalize the Autogrow payload ({'image_0': tensor, ...} / list / tensor)
    into a single batched IMAGE tensor, or None when nothing is connected."""
    if images is None:
        return None
    values = []
    if isinstance(images, dict):
        # sorted() is correct here because the Autogrow keys are image_0..image_3
        # (max 4 < 10), so lexicographic order == numeric order.
        for key in sorted(images.keys()):
            values.append(images[key])
    elif isinstance(images, (list, tuple)):
        values = list(images)
    else:
        values = [images]
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    import torch
    # Each Autogrow socket yields [1, H, W, C] and H/W may differ between
    # sockets, which makes torch.cat(dim=0) raise. Mirror ComfyUI core's
    # ImageBatch: center-crop every tensor (equally from both sides) to the
    # minimum (H, W) across the batch before concatenating.
    min_h = min(int(v.shape[1]) for v in values)
    min_w = min(int(v.shape[2]) for v in values)
    cropped = []
    for v in values:
        h, w = int(v.shape[1]), int(v.shape[2])
        top = (h - min_h) // 2
        left = (w - min_w) // 2
        cropped.append(v[:, top:top + min_h, left:left + min_w, :])
    return torch.cat(cropped, dim=0)


def _n_images(image_tensor) -> int:
    if image_tensor is None:
        return 0
    try:
        return int(image_tensor.shape[0])
    except Exception:
        return 1


class H3VisionPromptor(io.ComfyNode):
    """Local, API-key-free MiniMax H3 prompt enhancer powered by a native VLM
    text encoder (Qwen2.5-VL / Qwen3-VL / Gemma vision)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VisionPromptor",
            display_name="H3 Vision Promptor (Local VLM)",
            category=CATEGORY,
            search_aliases=["MiniMax", "H3", "prompt", "VLM", "vision prompt"],
            description=(
                "Turns an idea (+ optional reference images) into a production-ready MiniMax H3 "
                "prompt using a LOCAL VLM running on ComfyUI's native text-generation stack. "
                "No API keys, no external services."
            ),
            inputs=[
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional: connect CLIP from the native CLIPLoader node. "
                                      "Overrides the text_encoder dropdown."),
                io.Combo.Input("text_encoder", options=_text_encoder_options(),
                               tooltip="VLM checkpoint from models/text_encoders/ "
                                       "(e.g. a Qwen3-VL or Gemma-3 instruct repack)."),
                io.String.Input("user_idea", multiline=True, default="",
                                placeholder="A young woman on a night train reads a letter, then smiles..."),
                _images_autogrow(min_images=0),
                io.Combo.Input("task_type", options=TASK_OPTIONS, default="Auto",
                               tooltip="Auto: 0 images=T2VA, 1=I2VA, 2=FL2VA, 3+=Ref2VA. "
                                       "Pick L2VA explicitly for last-frame tasks."),
                io.Float.Input("duration", default=8.0, min=4.0, max=15.0, step=0.5,
                               tooltip="Target video length in seconds. In planner mode this is "
                                       "the TARGET length of one clip — used when "
                                       "clip_duration is 0 (auto)."),
                io.Combo.Input("vision_mode", options=_vision_preset_keys() + [SKIP_VISION],
                               tooltip="How the VLM describes the reference image(s) before writing. "
                                       "'skip (idea only)' writes from the idea alone."),
                io.String.Input("extra_instructions", multiline=True, default="", optional=True,
                                placeholder="Optional extras: dialogue lines, mood, camera wishes..."),
                io.String.Input("custom_system_prompt", multiline=True, default="", optional=True,
                                advanced=True,
                                tooltip="Advanced: fully replace the built-in H3 system prompt."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,
                             control_after_generate="randomize"),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=64, min=0, max=500),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=64),
                io.Int.Input("variants", default=1, min=1, max=4,
                             tooltip="Generate N prompt variants (seed, seed+1, ...) in one run."),
                io.Boolean.Input("use_default_template", default=False, advanced=True,
                                 tooltip="Let the model's built-in chat template wrap the prompt "
                                         "instead of this node's H3-aware template."),
                io.Boolean.Input("keep_model_loaded", default=True, advanced=True,
                                 tooltip="Cache the text-encoder CLIP between runs (faster, uses VRAM/RAM)."),
                # --- Planner-mode inputs (fork) --------------------------------
                # WIDGET-ORDER CONTRACT: ComfyUI renders required widgets first,
                # then optional ones, and saved workflows apply widgets_values
                # POSITIONALLY. To keep workflows saved with the upstream
                # original node loading byte-identically, the original widget
                # order must stay an exact prefix — so these two inputs MUST be
                # optional=True and MUST stay at the very END of this list.
                # (They then render at the bottom of the panel, after
                # extra_instructions / custom_system_prompt.) Inserting them in
                # the middle shifts seed/"randomize"/temperature/top_p/max_tokens
                # onto wrong widgets and triggers "Input not in range" errors.
                io.Boolean.Input("emit_planner_outputs", default=False, optional=True,
                                 tooltip="Planner mode: emit LongMedia Planner outputs "
                                         "(planner_prompt, planner_prompt_ru, global_prompt, "
                                         "clip_durations, camera_params, "
                                         "reference_images_status) and disable the original "
                                         "prompt/vision_context outputs. Mutually exclusive "
                                         "with the original mode."),
                io.Float.Input("total_duration", default=24.0, min=8.0, max=2400.0, step=1.0,
                               optional=True,
                               tooltip="Planner mode only: total video duration in seconds "
                                       "(clip_count = ceil(total / clip target), clamped to "
                                       "LongMedia's 2..16 clips)."),
                io.Float.Input("clip_duration", default=0.0, min=0.0, max=150.0, step=0.5,
                               optional=True,
                               tooltip="Planner mode only: TARGET duration of ONE clip, "
                                       "0 = auto (use the duration widget). Allows clip "
                                       "targets beyond the original node's 15s single-video "
                                       "cap — up to LongMedia's 150s per-clip limit. In "
                                       "planner mode it overrides 'duration'; in the "
                                       "original mode it is ignored."),
                io.Combo.Input("creative_boost",
                               options=list(planner_mode.CREATIVE_LEVELS),
                               default=planner_mode.CREATIVE_LEVEL_DEFAULT, optional=True,
                               tooltip="Planner mode only (v1.1.5): creative writing boost. "
                                       "'off' = byte-identical v1.1.4 prompts; 'standard' = "
                                       "vividness standards in the system prompt; "
                                       "'expressive' = standard + per-clip visual motif "
                                       "ingredients + denser word budgets. In the original "
                                       "mode it is ignored."),
                io.String.Input("style_directive", multiline=True, default="", optional=True,
                                placeholder="Planner mode only: your creative brief — genre, "
                                            "mood, visual style, references...",
                                tooltip="Planner mode only (v1.1.5): the author's brief — "
                                        "genre / mood / visual style / references, passed "
                                        "verbatim to the planner pass at any creative_boost "
                                        "level. Example: 'rain-soaked neon noir, melancholy, "
                                        "Wong Kar-wai colors'."),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("vision_context"),
                io.String.Output("debug"),
                io.String.Output("planner_prompt",
                                 tooltip="LongMedia Planner import text: clip_1: ... clip_N: "
                                         "sections, no camera language."),
                io.String.Output("planner_prompt_ru",
                                 tooltip="Russian mirror of the plan: GLOBAL: + clip_N: "
                                         "sections (dialogue <d> blocks verbatim)."),
                io.String.Output("global_prompt",
                                 tooltip="Constant scene description for the Planner's "
                                         "global_prompt input."),
                io.String.Output("clip_durations",
                                 tooltip="JSON array of per-clip durations, e.g. [8.0, 8.0, 8.0]."),
                io.String.Output("camera_params",
                                 tooltip="JSON array of LongMedia Cameras cards (13 keys per "
                                         "card, values from the Cameras vocabulary)."),
                io.String.Output("reference_images_status",
                                 tooltip="JSON marker of connected reference images "
                                         "(presence/count/slots; tensors carry no paths)."),
            ],
        )

    @classmethod
    def execute(cls, text_encoder, user_idea, task_type, duration, vision_mode,
                seed, temperature, top_p, top_k, max_tokens, variants,
                use_default_template, keep_model_loaded,
                clip=None, images=None, extra_instructions="",
                custom_system_prompt="", emit_planner_outputs=False,
                total_duration=24.0, clip_duration=0.0,
                creative_boost=planner_mode.CREATIVE_LEVEL_DEFAULT,
                style_directive="") -> io.NodeOutput:
        if emit_planner_outputs:
            return cls._execute_planner(
                text_encoder=text_encoder, user_idea=user_idea, task_type=task_type,
                duration=duration, vision_mode=vision_mode, seed=seed,
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_tokens=max_tokens, variants=variants,
                use_default_template=use_default_template,
                keep_model_loaded=keep_model_loaded, clip=clip, images=images,
                extra_instructions=extra_instructions,
                custom_system_prompt=custom_system_prompt,
                total_duration=total_duration, clip_duration=clip_duration,
                creative_boost=creative_boost, style_directive=style_directive)

        t_start = time.time()

        clip_obj, source_desc = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        image_tensor = _collect_images(images)
        n_images = _n_images(image_tensor)
        task = prompt_builder.detect_task(task_type, n_images)

        # --- vision pass -------------------------------------------------
        vision_context = ""
        vision_seconds = 0.0
        vision_error = None
        if image_tensor is not None and vision_mode != SKIP_VISION:
            t_vis = time.time()
            try:
                vision_context = vision.analyze_images(
                    clip_obj, image_tensor, vision_mode, None,
                    seed=seed, temperature=temperature, max_tokens=max_tokens,
                )
            except Exception as e:
                vision_error = str(e)
                print(f"[H3 VisionPromptor] Vision pass failed, continuing without it: {e}")
            vision_seconds = time.time() - t_vis

        # --- prompt-writing generation -----------------------------------
        system, user = prompt_builder.build_messages(
            task, duration, user_idea, vision_context, extra_instructions,
            "English", custom_system_prompt,
        )

        results: list[str] = []
        warnings: list[str] = []
        t_gen = time.time()
        for i in range(int(variants)):
            raw = engine.generate_text(
                clip_obj, system, user,
                image_tensor=image_tensor,
                seed=int(seed) + i,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                use_default_template=use_default_template,
            )
            clean, warns = post_processor.finalize(raw, task, duration)
            results.append(clean)
            warnings.extend(f"[variant {i + 1}] {w}" for w in warns)
        gen_seconds = time.time() - t_gen

        if vision_error:
            warnings.append(f"vision pass failed: {vision_error}")

        prompt = results[0]
        for i in range(1, len(results)):
            prompt += f"\n\n===== VARIANT {i + 1} =====\n\n" + results[i]

        debug = json.dumps({
            "clip_source": source_desc,
            "model_family": engine.detect_family(clip_obj),
            "detected_task": task,
            "n_images": n_images,
            "duration_seconds": duration,
            "word_budget": prompt_builder.word_budget(task, duration),
            "vision_mode": vision_mode if image_tensor is not None else f"{vision_mode} (no images)",
            "vision_seconds": round(vision_seconds, 2),
            "generation_seconds": round(gen_seconds, 2),
            "total_seconds": round(time.time() - t_start, 2),
            "variants": int(variants),
            "warnings": warnings,
        }, indent=2, ensure_ascii=False)

        return io.NodeOutput(prompt, vision_context, debug,
                             "", "", "", "", "", "")

    # ------------------------------------------------------------------
    # Planner mode (fork): LongMedia Planner + Cameras outputs.
    # ------------------------------------------------------------------

    @classmethod
    def _execute_planner(cls, text_encoder, user_idea, task_type, duration, vision_mode,
                         seed, temperature, top_p, top_k, max_tokens, variants,
                         use_default_template, keep_model_loaded, clip, images,
                         extra_instructions, custom_system_prompt,
                         total_duration, clip_duration=0.0,
                         creative_boost=planner_mode.CREATIVE_LEVEL_DEFAULT,
                         style_directive="") -> io.NodeOutput:
        t_start = time.time()
        warnings: list[str] = []

        # Creative boost level (v1.1.5): normalize early — an unknown combo
        # value must degrade to 'standard' with a warning, never crash the
        # planner pipeline.
        level, level_warning = planner_mode.normalize_creative_boost(creative_boost)
        if level_warning:
            warnings.append(level_warning)
        style_directive = str(style_directive or "")

        # Effective per-clip target: the planner-only clip_duration widget
        # (0 = auto) overrides the original 4-15s duration widget, so planner
        # plans may target clips up to LongMedia's 150s per-clip limit.
        try:
            clip_duration_val = float(clip_duration or 0.0)
        except (TypeError, ValueError):
            clip_duration_val = 0.0
        if clip_duration_val > 0:
            target_clip_duration = clip_duration_val
            target_source = "clip_duration"
        else:
            target_clip_duration = float(duration)
            target_source = "duration"

        clip_obj, source_desc = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        # Fail fast on MiniMax conditioning encoders (the H3 text-encoder
        # file): they cannot chat-generate at all — letting them through
        # wastes minutes on degenerate punctuation output (v1.1.4).
        family = engine.assert_chat_capable(clip_obj)
        image_tensor = _collect_images(images)
        n_images = _n_images(image_tensor)
        task = prompt_builder.detect_task(task_type, n_images)  # informational only

        if int(variants) != 1:
            warnings.append(f"variants={int(variants)} ignored in planner mode (forced to 1; "
                            "multiple variants would break the Planner import format)")

        if family == "generic":
            warnings.append(
                "model_family='generic': the prompt was sent as RAW text with no chat "
                "template — if the selected model is actually a Qwen/Gemma instruct "
                "repack the fork failed to recognize it; check family_signals in this "
                "debug output and report it"
            )

        # --- vision pass (same stack as the original mode + thinking
        # suppression for qwen models — a Qwen3 vision pass left in thinking
        # mode burns its tokens before writing a single description line) ----
        vision_context = ""
        vision_seconds = 0.0
        if image_tensor is not None and vision_mode != SKIP_VISION:
            t_vis = time.time()
            try:
                vision_context = vision.analyze_images(
                    clip_obj, image_tensor, vision_mode, None,
                    seed=seed, temperature=temperature, max_tokens=max_tokens,
                    suppress_thinking=True,
                )
            except Exception as e:
                warnings.append(f"vision pass failed, continuing without it: {e}")
                print(f"[H3 VisionPromptor] Vision pass failed, continuing without it: {e}")
            vision_seconds = time.time() - t_vis

        # --- duration arithmetic (deterministic, not VLM) -----------------
        clip_count, count_warnings = planner_mode.planned_clip_count(total_duration, target_clip_duration)
        warnings.extend(count_warnings)
        durations, dur_warnings = planner_mode.compute_clip_durations(
            total_duration, target_clip_duration, clip_count)
        warnings.extend(dur_warnings)

        # --- single planner generation pass ------------------------------
        # Expressive level: per-clip motif ingredients, sampled
        # deterministically from the generation seed (same seed -> same
        # ingredients; new seed -> a new creative direction).
        motif_hints = (planner_mode.sample_motif_hints(clip_count, int(seed))
                       if level == "expressive" else None)
        system, user = prompt_builder.build_planner_messages(
            total_duration, target_clip_duration, clip_count, durations,
            user_idea, vision_context, extra_instructions,
            custom_system_prompt, n_images,
            style_directive=style_directive, creative_boost=level,
            motif_hints=motif_hints,
        )
        planner_budget = planner_mode.estimate_planner_tokens(
            clip_count, durations, creative_boost=level)
        t_gen = time.time()
        raw = engine.generate_text(
            clip_obj, system, user,
            image_tensor=image_tensor,
            seed=int(seed),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=planner_budget,
            use_default_template=use_default_template,
            suppress_thinking=True,
        )
        planner_seconds = time.time() - t_gen

        split = planner_mode.split_plan(raw)
        warnings.extend(split.warnings)

        clips, clip_warnings = planner_mode.parse_clips(split.clips_text, clip_count)
        warnings.extend(clip_warnings)

        # Reconcile the arithmetic plan to the ACTUAL clip count the model
        # produced (2..16), keeping the total duration intact.
        if clips and len(clips) != clip_count and planner_mode.MIN_CLIPS <= len(clips) <= planner_mode.MAX_CLIPS:
            durations = planner_mode.compute_clip_durations(
                total_duration, total_duration / len(clips), len(clips))[0]
            clip_count = len(clips)

        # --- photo-meta leak scrub (v1.1.7) --------------------------------
        # The contract forbids describing the source photograph, but a small
        # instruct-VLM can still copy the reference-image analysis into the
        # plan ("The photo shows ..." -> RU mirror "На фото изображено ...").
        # Deterministic backstop: tier-1 (explicit source references) is
        # stripped everywhere; tier-2 (generic photo wording) is stripped
        # only from GLOBAL — inside clips a photograph may be a diegetic
        # prop, so there it only warns. Clean plans pass through untouched.
        photo_meta_removed = 0
        if clips:
            scrubbed_clips = []
            for i, body in enumerate(clips, start=1):
                clean, scrub_w, scrub_n = planner_mode.scrub_photo_meta(
                    body, strict=False, label=f"clip_{i}")
                if body.strip() and not clean.strip():
                    warnings.append(
                        f"photo-meta scrub emptied clip_{i} (v1.1.7) — the whole clip "
                        "was photo-description; re-seed the node or write a richer idea")
                scrubbed_clips.append(clean)
                warnings.extend(scrub_w)
                photo_meta_removed += scrub_n
            clips = scrubbed_clips

        planner_prompt = planner_mode.assemble_planner_prompt(clips) if clips else ""
        global_prompt = split.global_text
        if not split.has_global:
            global_prompt = ""
        elif global_prompt.strip():
            global_prompt, g_scrub_w, g_scrub_n = planner_mode.scrub_photo_meta(
                global_prompt, strict=True, label="GLOBAL")
            warnings.extend(g_scrub_w)
            photo_meta_removed += g_scrub_n
        if not clips:
            warnings.append(
                "planner answer yielded no parseable clip sections — "
                "planner_prompt / planner_prompt_ru / global_prompt are empty; "
                "inspect raw_planner_answer in this debug output to see what the "
                "model actually returned"
            )
        elif not global_prompt.strip():
            # LongMedia joins global_prompt with EVERY clip prompt at runtime
            # (_v043_join_global_local_prompt); an empty global means the
            # constants (identity / environment / style) never reach the
            # Planner and the clips render as independent scenes (v1.1.6).
            warnings.append(
                "GLOBAL section is missing or empty — scene constants (subject "
                "identity, environment, style) will NOT reach the LongMedia "
                "Planner, and clip prompts alone read as independent scenes. "
                "Inspect raw_planner_answer in this debug output, retry with a "
                "different seed, and make sure global_prompt is wired to the "
                "Planner's global_prompt input."
            )

        cards, cam_warnings = planner_mode.parse_cameras(
            split.cameras_text, len(clips) if clips else clip_count)
        warnings.extend(cam_warnings)
        camera_params = planner_mode.cameras_to_json(cards)

        # --- translation pass (EN -> RU mirror) ---------------------------
        planner_prompt_ru = ""
        translate_seconds = 0.0
        translate_budget = 0
        raw_ru = ""
        if clips:
            t_tr = time.time()
            try:
                tsys, tuser = prompt_builder.build_translate_messages(
                    planner_mode.translate_source(global_prompt, clips),
                    preserve_vividness=(level != "off"))
                translate_budget = planner_mode.estimate_translate_tokens(
                    planner_mode.translate_source(global_prompt, clips))
                raw_ru = engine.generate_text(
                    clip_obj, tsys, tuser,
                    image_tensor=None,
                    seed=int(seed) + 1,
                    temperature=TRANSLATE_TEMPERATURE,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=translate_budget,
                    use_default_template=use_default_template,
                    suppress_thinking=True,
                )
                planner_prompt_ru, tr_warnings = planner_mode.parse_translated(raw_ru)
                warnings.extend(tr_warnings)
                # RU-mirror backstop (v1.1.7): the translator only mirrors
                # the (already scrubbed) EN text, but scrub tier-1 anyway so
                # a stray "На фото изображено ..." can never reach the reader.
                ru_clean, ru_w, ru_n = planner_mode.scrub_photo_meta(
                    planner_prompt_ru, strict=False, label="RU mirror")
                if ru_w or ru_n:
                    planner_prompt_ru = ru_clean
                    warnings.extend(ru_w)
                    photo_meta_removed += ru_n
            except Exception as e:
                warnings.append(f"translation pass failed: {e}")
                print(f"[H3 VisionPromptor] Translation pass failed: {e}")
            translate_seconds = time.time() - t_tr

        reference_status = planner_mode.reference_images_status(images)

        debug = json.dumps({
            "mode": "planner",
            "clip_source": source_desc,
            "model_family": family,
            "detected_task": task,
            "n_images": n_images,
            "reference_images_status": json.loads(reference_status),
            "total_duration": float(total_duration),
            "target_clip_duration": float(target_clip_duration),
            "target_clip_duration_source": target_source,
            # Creative boost (v1.1.5): what was actually injected after
            # normalization (the widget value is echoed in the warning above
            # when it was unknown).
            "creative_boost": level,
            "style_directive": (style_directive.strip()[:240] or None),
            "motif_hints": ({"world": list(motif_hints.get("world") or ()),
                             "clips": {str(k): list(v) for k, v in (motif_hints.get("clips") or {}).items()}}
                            if motif_hints else None),
            "global_prompt_chars": len(global_prompt),
            # Photo-meta scrubber (v1.1.7): how many leaked photo-description
            # sentences were removed from GLOBAL / clips / the RU mirror.
            "photo_meta_scrubbed": photo_meta_removed,
            "clip_count": int(clip_count) if clips else 0,
            "clips_parsed": len(clips),
            "clip_durations": durations,
            "cameras_validated": len(cards),
            "camera_vocab": camera_knowledge.enum_sizes(),
            "planner_token_budget": planner_budget,
            "translate_token_budget": translate_budget,
            "use_default_template": bool(use_default_template),
            # Honest flag (v1.1.4): the think suppressor is only APPLIED for
            # family='qwen' (engine.build_chat_text) — report the effective
            # state, not the requested one, so a generic-family run can no
            # longer masquerade as suppressed in the debug output.
            "thinking_suppressed": family == "qwen",
            "chat_format": {"qwen": "chatml", "gemma4": "gemma4-turns",
                            "gemma": "gemma-turns"}.get(family, "raw (no chat template)"),
            "family_signals": engine.family_signals_text(clip_obj),
            # Raw model answers (BEFORE light_clean strips think-blocks), so a
            # degenerate generation is diagnosable from a Previewer on `debug`
            # instead of guessing from empty planner outputs.
            "raw_planner_answer": str(raw)[:1600],
            "raw_planner_answer_chars": len(str(raw)),
            "raw_translate_answer": (str(raw_ru)[:1600] if raw_ru else None),
            "vision_seconds": round(vision_seconds, 2),
            "planner_seconds": round(planner_seconds, 2),
            "translate_seconds": round(translate_seconds, 2),
            "total_seconds": round(time.time() - t_start, 2),
            "warnings": warnings,
        }, indent=2, ensure_ascii=False)

        return io.NodeOutput("", "", debug,
                             planner_prompt, planner_prompt_ru, global_prompt,
                             json.dumps(durations), camera_params, reference_status)


class H3VisionAnalyzer(io.ComfyNode):
    """Standalone VLM image analysis (one batched vision pass)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VisionAnalyzer",
            display_name="H3 Vision Analyzer (Local VLM)",
            category=CATEGORY,
            search_aliases=["MiniMax", "H3", "vision", "describe", "VLM"],
            description=(
                "Describes reference image(s) with a LOCAL VLM (native text generation). "
                "Feed the description into the H3 Vision Promptor or use it standalone."
            ),
            inputs=[
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional: connect CLIP from the native CLIPLoader node."),
                io.Combo.Input("text_encoder", options=_text_encoder_options(),
                               tooltip="VLM checkpoint from models/text_encoders/."),
                _images_autogrow(min_images=1),
                io.Combo.Input("vision_mode", options=_vision_preset_keys() + [CUSTOM_VISION],
                               tooltip="Analysis preset, or 'custom' to use custom_prompt."),
                io.String.Input("custom_prompt", multiline=True, default="", optional=True,
                                placeholder="Used when vision_mode = custom."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,
                             control_after_generate="randomize"),
                io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.05),
                io.Int.Input("max_tokens", default=2048, min=256, max=8192, step=64),
                io.Boolean.Input("keep_model_loaded", default=True, advanced=True,
                                 tooltip="Cache the text-encoder CLIP between runs."),
            ],
            outputs=[
                io.String.Output("description"),
            ],
        )

    @classmethod
    def execute(cls, text_encoder, vision_mode, seed, temperature, max_tokens,
                keep_model_loaded, clip=None, images=None, custom_prompt="") -> io.NodeOutput:
        clip_obj, _source = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        image_tensor = _collect_images(images)
        if image_tensor is None:
            raise RuntimeError(
                "[H3 VisionPromptor] H3VisionAnalyzer needs at least one image: "
                "connect an IMAGE output to the node's image input(s)."
            )
        prompt_override = (custom_prompt or "").strip() if vision_mode == CUSTOM_VISION else None
        if vision_mode == CUSTOM_VISION and not prompt_override:
            raise RuntimeError(
                "[H3 VisionPromptor] vision_mode='custom' requires a non-empty custom_prompt: "
                "type your analysis instruction into the custom_prompt input."
            )
        description = vision.analyze_images(
            clip_obj, image_tensor, vision_mode, prompt_override,
            seed=seed, temperature=temperature, max_tokens=max_tokens,
        )
        return io.NodeOutput(description)
