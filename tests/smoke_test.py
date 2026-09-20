#!/usr/bin/env python3
"""Smoke test for ComfyUI-H3-VisionPromptor-Planner — runs OUTSIDE ComfyUI.

Stubs comfy_api.latest / folder_paths / comfy.sd with fakes, then verifies:
  1. the extension package imports and exposes comfy_entrypoint
  2. both node schemas define with the expected ids / inputs / outputs
     (incl. the fork's emit_planner_outputs / total_duration inputs and the
     six planner outputs)
  3. post_processor.finalize cleans a messy sample (fences + think block +
     preamble + missing audio fields + bad timestamps)
  4. prompt_builder.detect_task matrix (Auto mapping + explicit overrides)
  5. engine.generate_text end-to-end against a fake generative CLIP
  6. gemma4 family detection + template shape (media block AFTER the text)
  7. gemma (Gemma-3) emits exactly ONE <image_soft_token> regardless of n_images
  8. _collect_images center-crops mixed-resolution batches to min (H, W)
  9. vision_mode='custom' with an empty custom_prompt raises RuntimeError
  10. finalize trims BEFORE appending audio fields (appended fields survive)
  11. planner mode end-to-end on a scripted fake CLIP: two generation passes
     (plan + translation), 9 outputs, planner JSONs well-formed, variants
     forced to 1, token budgets requested independently of the widget
  12. original mode is behavior-identical to the upstream package (same
     prompt/vision_context output; same debug keys modulo timings)
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

PKG_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Stubs for ComfyUI modules
# ---------------------------------------------------------------------------

def _make_io_type(name):
    class Input:
        def __init__(self, id=None, **kwargs):
            self.id = id
            self.kwargs = kwargs

    class Output:
        def __init__(self, id=None, **kwargs):
            self.id = id
            self.kwargs = kwargs

    Input.__qualname__ = f"{name}.Input"
    Output.__qualname__ = f"{name}.Output"
    return type(name, (), {"Input": Input, "Output": Output, "Type": object})


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    def result(self):
        return self.args or None


class _ComfyNode:
    @classmethod
    def define_schema(cls):
        raise NotImplementedError


class _ComfyExtension:
    async def get_node_list(self):
        return []


def _install_stubs():
    io = types.ModuleType("comfy_api.latest.io")
    for name in ("Clip", "Combo", "String", "Float", "Int", "Boolean", "Image",
                 "Audio", "Video", "Mask", "Latent", "Conditioning"):
        setattr(io, name, _make_io_type(name))

    class _AutogrowTemplate:
        def __init__(self, input=None, **kwargs):
            self.input = input
            self.kwargs = kwargs

    class _AutogrowInput:
        def __init__(self, id=None, template=None, **kwargs):
            self.id = id
            self.template = template
            self.kwargs = kwargs

    io.Autogrow = type("Autogrow", (), {
        "Input": _AutogrowInput,
        "TemplatePrefix": _AutogrowTemplate,
        "TemplateNames": _AutogrowTemplate,
        "Type": dict,
    })
    io.Schema = _Schema
    io.NodeOutput = _NodeOutput
    io.ComfyNode = _ComfyNode

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ComfyExtension = _ComfyExtension

    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda folder: ["fake_qwen_vl.safetensors"] if folder == "text_encoders" else []
    folder_paths.get_full_path_or_raise = lambda folder, name: f"/fake/models/{folder}/{name}"
    folder_paths.get_folder_paths = lambda folder: [f"/fake/models/{folder}"]

    class _FakeTransformer:
        pass

    _FakeTransformer.__name__ = "Qwen2_5_VLTransformer"

    class FakeTokenizer:
        pass

    FakeTokenizer.__name__ = "Qwen2Tokenizer"

    class FakeClip:
        """Mimics the native generative CLIP interface."""
        def __init__(self):
            self.cond_stage_model = types.SimpleNamespace(transformer=_FakeTransformer())
            self.tokenizer = FakeTokenizer()
            self.calls = []

        def tokenize(self, prompt, image=None, skip_template=False, min_length=1):
            self.calls.append(("tokenize", prompt, image, skip_template, min_length))
            return {"prompt": prompt, "image": image}

        def generate(self, tokens, do_sample=False, max_length=512, temperature=1.0,
                     top_k=50, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
                     presence_penalty=0.0, seed=None):
            self.calls.append(("generate", max_length, seed))
            return [1, 2, 3]

        def decode(self, ids):
            return ("integrated_multimodal_description: [Shot 1] Cinematic, a wide shot "
                    "frames a quiet platform.\n\noverall_soundscape: Distant rail hum.\n\n"
                    "non_diegetic_music: N/A")

    sd = types.ModuleType("comfy.sd")
    sd.load_clip = lambda ckpt_paths, embedding_directory=None, clip_type=None: FakeClip()
    sd.CLIPType = types.SimpleNamespace(STABLE_DIFFUSION="stable_diffusion")

    comfy = types.ModuleType("comfy")
    comfy.sd = sd

    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = io
    sys.modules["folder_paths"] = folder_paths
    sys.modules["comfy"] = comfy
    sys.modules["comfy.sd"] = sd
    return FakeClip


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "h3_vision_promptor_pkg", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main():
    FakeClip = _install_stubs()
    pkg = _load_package()

    # 1. extension import + entrypoint
    check("extension exposes comfy_entrypoint", hasattr(pkg, "comfy_entrypoint"))
    ext = asyncio.run(pkg.comfy_entrypoint())
    node_list = asyncio.run(ext.get_node_list())
    check("extension node list has both nodes", len(node_list) == 2)

    from h3_vision_promptor_pkg.py import nodes, post_processor, prompt_builder, engine, vision

    # 2. schemas
    s1 = nodes.H3VisionPromptor.define_schema()
    check("H3VisionPromptor node_id", s1.node_id == "H3VisionPromptor")
    check("H3VisionPromptor display_name", s1.display_name == "H3 Vision Promptor (Local VLM)")
    check("H3VisionPromptor category", s1.category == "H3/Promptor")
    expected_inputs = ["clip", "text_encoder", "user_idea", "images", "task_type",
                       "duration", "vision_mode", "extra_instructions",
                       "custom_system_prompt", "seed", "temperature", "top_p",
                       "top_k", "max_tokens", "variants", "use_default_template",
                       "keep_model_loaded", "emit_planner_outputs", "total_duration",
                       "clip_duration", "creative_boost", "style_directive"]
    got_inputs = [i.id for i in s1.inputs]
    check("H3VisionPromptor input order/names", got_inputs == expected_inputs, f"got {got_inputs}")
    check("H3VisionPromptor outputs",
          [o.id for o in s1.outputs] == ["prompt", "vision_context", "debug",
                                         "planner_prompt", "planner_prompt_ru",
                                         "global_prompt", "clip_durations",
                                         "camera_params", "reference_images_status"])
    seed_in = s1.inputs[9]
    check("seed control_after_generate=randomize",
          seed_in.kwargs.get("control_after_generate") == "randomize")
    # Widget-order contract: the original node's widget order must be an exact
    # prefix (required first, optional after) so old workflows load cleanly.
    orig_prefix = [i.id for i in s1.inputs[:17]]
    check("original widget order kept as prefix",
          orig_prefix == ["clip", "text_encoder", "user_idea", "images", "task_type",
                          "duration", "vision_mode", "extra_instructions",
                          "custom_system_prompt", "seed", "temperature", "top_p",
                          "top_k", "max_tokens", "variants", "use_default_template",
                          "keep_model_loaded"], f"got {orig_prefix}")
    emit_in = s1.inputs[17]
    total_in = s1.inputs[18]
    clipdur_in = s1.inputs[19]
    boost_in = s1.inputs[20]
    style_in = s1.inputs[21]
    check("emit_planner_outputs optional (renders after optionals)",
          bool(emit_in.kwargs.get("optional")) is True)
    check("total_duration optional (renders after optionals)",
          bool(total_in.kwargs.get("optional")) is True)
    check("clip_duration optional, max=150 (LongMedia per-clip cap)",
          bool(clipdur_in.kwargs.get("optional")) is True
          and clipdur_in.kwargs.get("max") == 150.0
          and clipdur_in.kwargs.get("default") == 0.0)
    check("creative_boost combo optional with the three levels (v1.1.5)",
          bool(boost_in.kwargs.get("optional")) is True
          and list(boost_in.kwargs.get("options"))
          == ["off", "standard", "expressive"]
          and boost_in.kwargs.get("default") == "standard")
    check("style_directive optional multiline string (v1.1.5)",
          bool(style_in.kwargs.get("optional")) is True
          and bool(style_in.kwargs.get("multiline")) is True
          and style_in.kwargs.get("default") == "")
    # Regression (v1.1.1): a workflow saved with the ORIGINAL node stores
    # widgets_values POSITIONALLY (16 slots). The frontend renders widgets as
    # required-inputs (list order) followed by optional-inputs (list order),
    # and the seed widget takes TWO slots (value + control_after_generate).
    # Simulate that ordering from the fork schema and verify every old slot
    # would land on the identically-named widget.
    widget_only = [i for i in s1.inputs if i.id not in ("clip", "images")]
    req = [i.id for i in widget_only if not i.kwargs.get("optional")]
    opt = [i.id for i in widget_only if i.kwargs.get("optional")]
    slots = []
    for name in req + opt:
        slots.append(name)
        if name == "seed":
            slots.append("<control_after_generate slot>")
    orig_slots = slots[:-5]  # fork adds emit/total/clip_duration/boost/style at the tail
    expect_orig = ["text_encoder", "user_idea", "task_type", "duration", "vision_mode",
                   "seed", "<control_after_generate slot>", "temperature", "top_p", "top_k",
                   "max_tokens", "variants", "use_default_template", "keep_model_loaded",
                   "extra_instructions", "custom_system_prompt"]
    check("old 16-slot widgets_values lands on same-named widgets",
          orig_slots == expect_orig, f"got {orig_slots}")
    check("v1.1.2 19-slot widgets_values lands on same-named widgets (prefix kept)",
          slots[:-2] == expect_orig + ["emit_planner_outputs", "total_duration", "clip_duration"],
          f"got {slots[:-2]}")
    check("fork widget slots = original + 5 planner widgets",
          slots == expect_orig + ["emit_planner_outputs", "total_duration", "clip_duration",
                                  "creative_boost", "style_directive"],
          f"got {slots}")
    imgs_in = s1.inputs[3]
    check("images autogrow template prefix image_", imgs_in.template.kwargs.get("prefix") == "image_")
    check("images autogrow min=0 max=4",
          imgs_in.template.kwargs.get("min") == 0 and imgs_in.template.kwargs.get("max") == 4)

    s2 = nodes.H3VisionAnalyzer.define_schema()
    check("H3VisionAnalyzer node_id", s2.node_id == "H3VisionAnalyzer")
    check("H3VisionAnalyzer display_name", s2.display_name == "H3 Vision Analyzer (Local VLM)")
    check("H3VisionAnalyzer output", [o.id for o in s2.outputs] == ["description"])
    a_inputs = [i.id for i in s2.inputs]
    check("H3VisionAnalyzer inputs",
          a_inputs == ["clip", "text_encoder", "images", "vision_mode", "custom_prompt",
                       "seed", "temperature", "max_tokens", "keep_model_loaded"],
          f"got {a_inputs}")
    check("H3VisionAnalyzer images min=1 max=4",
          s2.inputs[2].template.kwargs.get("min") == 1 and s2.inputs[2].template.kwargs.get("max") == 4)

    # 3. post_processor.finalize on a messy sample
    messy = (
        "Sure! Here is your rewritten prompt:\n\n"
        "```text\n"
        "<think>let me plan the shots carefully</think>\n"
        "integrated_multimodal_description: [Shot 1] Cinematic, a wide shot frames a quiet "
        "station platform at dawn. [Shot 2] At 00:09.000, the camera cuts to a close-up of a "
        "torn ticket. [Shot 3] At 00:04.000, the shot cuts to a departing train.\n"
        "```\n"
        "Hope this helps!"
    )
    clean, warns = post_processor.finalize(messy, "T2VA", 6.0)
    check("finalize strips preamble/fences/think",
          clean.startswith("integrated_multimodal_description:"))
    check("finalize removes trailing fence text", "Hope this helps" not in clean and "```" not in clean)
    check("finalize auto-appends overall_soundscape N/A", "overall_soundscape: N/A" in clean)
    check("finalize auto-appends non_diegetic_music N/A", "non_diegetic_music: N/A" in clean)
    check("finalize warns auto-append soundscape",
          any("auto-appended missing field: overall_soundscape:" in w for w in warns))
    check("finalize warns timestamps not increasing",
          any("not strictly increasing" in w for w in warns))
    check("finalize warns timestamp exceeds duration",
          any("exceeds video duration" in w for w in warns))

    # ensure_alignment
    i2va_clean, _ = post_processor.finalize(
        "integrated_multimodal_description: [Shot 1] Cinematic...\n\n"
        "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A", "I2VA", 8.0)
    check("ensure_alignment prepends I2VA line",
          i2va_clean.startswith("For the target video, at 0.00 seconds into the target video, "
                                "<Picture 1> (from [Shot 1]) is fully referenced."))
    fl2va_clean, _ = post_processor.finalize(
        "integrated_multimodal_description: [Shot 1] Cinematic...\n\n"
        "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A", "FL2VA", 8.0)
    check("ensure_alignment prepends FL2VA line with duration %.2f",
          fl2va_clean.startswith("How the reference pictures align with the target video")
          and "8.00-second mark" in fl2va_clean)

    # smart trim
    long_text = "integrated_multimodal_description: " + ("word\n" * 2000) + \
                "\n\noverall_soundscape: x\n\nnon_diegetic_music: N/A"
    trimmed, twarns = post_processor.finalize(long_text, "T2VA", 8.0)
    check("finalize trims to <=7000 chars", len(trimmed) <= 7000)

    # m1: trim happens BEFORE appending missing audio fields, so the appended
    # fields always survive (finalize only warns about the final length).
    long_missing_audio = "integrated_multimodal_description: " + ("word\n" * 2000)
    trimmed2, twarns2 = post_processor.finalize(long_missing_audio, "T2VA", 8.0)
    check("finalize keeps appended audio fields after trim",
          trimmed2.rstrip().endswith("non_diegetic_music: N/A")
          and "overall_soundscape: N/A" in trimmed2)
    check("finalize reports auto-append after trim",
          any("auto-appended missing field: overall_soundscape:" in w for w in twarns2)
          and any("auto-appended missing field: non_diegetic_music:" in w for w in twarns2))
    check("finalize warns about trim without stale length claim",
          any("trimmed at the last line boundary" in w for w in twarns2))

    # 4. task detection matrix
    matrix = {(0, "Auto"): "T2VA", (1, "Auto"): "I2VA", (2, "Auto"): "FL2VA",
              (3, "Auto"): "Ref2VA", (4, "Auto"): "Ref2VA",
              (1, "L2VA"): "L2VA", (0, "t2va"): "T2VA", (2, "Ref2VA"): "Ref2VA"}
    ok = all(prompt_builder.detect_task(t, n) == want for (n, t), want in matrix.items())
    check("detect_task matrix", ok)
    try:
        prompt_builder.detect_task("BOGUS", 1)
        check("detect_task rejects unknown", False)
    except ValueError:
        check("detect_task rejects unknown", True)

    # word budgets
    check("word_budget base = duration*25", prompt_builder.word_budget("I2VA", 8.0) == 200)
    check("word_budget Ref2VA capped 500", prompt_builder.word_budget("Ref2VA", 20.0) == 500)

    # build_messages
    system, user = prompt_builder.build_messages("I2VA", 8.0, "a cat naps", "A grey cat on a sofa.",
                                                 "keep it calm", "English", "")
    check("build_messages loads base system prompt", "You are an H3 Prompt Writer" in system)
    check("build_messages includes task addendum", "I2VA" in system and "fully referenced" in system)
    check("build_messages embeds duration constraint",
          "Constraint: The video will be 8.0 seconds long (approx. 192 frames)." in user)
    check("build_messages embeds vision context + idea",
          "A grey cat on a sofa." in user and "a cat naps" in user)

    # 5. engine.generate_text with a fake generative CLIP
    clip = FakeClip()
    out = engine.generate_text(clip, "SYS", "USER", image_tensor=None, seed=42,
                               temperature=0.7, max_tokens=256)
    check("generate_text returns decoded text", out.startswith("integrated_multimodal_description:"))
    tok_call = clip.calls[0]
    check("tokenize called with native kwargs",
          tok_call[0] == "tokenize" and tok_call[3] is True and tok_call[4] == 1)
    check("family detected as qwen", engine.detect_family(clip) == "qwen")
    qtext = engine.build_chat_text("qwen", "SYS", "USER", 2, None)
    check("qwen chat template",
          qtext.startswith("<|im_start|>system\nSYS<|im_end|>")
          and qtext.count("<|vision_start|><|image_pad|><|vision_end|>") == 2
          and qtext.endswith("<|im_start|>assistant\n"))
    gtext = engine.build_chat_text("gemma", "SYS", "USER", 1, None)
    check("gemma chat template folds system into user turn",
          gtext.startswith("<start_of_turn>user\n<image_soft_token>\nSYS\n\nUSER")
          and gtext.endswith("<start_of_turn>model\n"))
    # m5: Gemma-3 binds only the first image token — exactly ONE soft token
    # no matter how many images are connected.
    gtext3 = engine.build_chat_text("gemma", "SYS", "USER", 3, None)
    check("gemma emits exactly ONE <image_soft_token> for n_images=3",
          gtext3.count("<image_soft_token>") == 1,
          f"got {gtext3.count('<image_soft_token>')}")
    gtext0 = engine.build_chat_text("gemma", "SYS", "USER", 0, None)
    check("gemma emits no soft token for n_images=0",
          "<image_soft_token>" not in gtext0)

    # M2: gemma4 family — detection (gemma4 must win over the 'gemma' substring)
    # and template shape (media block AFTER the text, one pair per image).
    class _G4Transformer:
        pass

    _G4Transformer.__name__ = "Gemma4Transformer"

    class _G4Tokenizer:
        pass

    _G4Tokenizer.__name__ = "Gemma4UnifiedTokenizer"

    g4clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(transformer=_G4Transformer()),
        tokenizer=_G4Tokenizer(),
    )
    check("detect_family returns gemma4 for Gemma-4 class names",
          engine.detect_family(g4clip) == "gemma4")
    g4text = engine.build_chat_text("gemma4", "SYS", "USER", 2, None)
    check("gemma4 template media after text, one pair per image",
          g4text == "<|turn>user\nSYS\n\nUSER\n<|image><|image|><|image><|image|>\n<|turn>model\n",
          f"got {g4text!r}")
    g4text0 = engine.build_chat_text("gemma4", "SYS", "USER", 0, None)
    check("gemma4 template without images has no media block",
          g4text0 == "<|turn>user\nSYS\n\nUSER\n<|turn>model\n", f"got {g4text0!r}")
    check("_strip_echo handles <|turn> terminator",
          engine._strip_echo("<|turn>user\nSYS\n\nUSER\n<|turn>model\nANSWER<|turn>", g4text0, "gemma4")
          == "ANSWER")
    otext = engine.build_chat_text("generic", "SYS", "USER", 0, "{system} || {user} || {images}")
    check("template override placeholders", otext == "SYS || USER || ")

    # resolve_clip prefers connected input and caches loader path
    c2, desc = engine.resolve_clip(clip, "<none - use CLIP input>")
    check("resolve_clip prefers clip input", c2 is clip and "connected" in desc)
    c3, desc3 = engine.resolve_clip(None, "fake_qwen_vl.safetensors")
    c4, _ = engine.resolve_clip(None, "fake_qwen_vl.safetensors")
    check("resolve_clip loads + caches text encoder", c3 is c4 and "text_encoders/" in desc3)

    # vision presets load
    presets = vision.load_vision_presets()
    check("vision presets loaded", "detailed_subject_scene" in presets and "scene_only" in presets)

    # m2: vision_mode='custom' with an empty/whitespace custom_prompt must raise.
    class FakeImage:
        shape = (1, 64, 64, 3)

    for bad_prompt in (None, "", "   \n  "):
        try:
            vision.analyze_images(clip, FakeImage(), "custom", bad_prompt,
                                  seed=1, temperature=0.2, max_tokens=64)
            check(f"custom vision mode rejects empty custom_prompt ({bad_prompt!r})", False)
        except RuntimeError as e:
            check(f"custom vision mode rejects empty custom_prompt ({bad_prompt!r})",
                  "requires a non-empty custom_prompt" in str(e), str(e))
    # analyzer node path raises the same clear error before touching the VLM
    try:
        nodes.H3VisionAnalyzer.execute(
            text_encoder="fake_qwen_vl.safetensors", vision_mode="custom",
            seed=1, temperature=0.2, max_tokens=256, keep_model_loaded=True,
            images={"image_0": FakeImage()}, custom_prompt="  ",
        )
        check("analyzer custom mode empty prompt raises", False)
    except RuntimeError as e:
        check("analyzer custom mode empty prompt raises",
              "requires a non-empty custom_prompt" in str(e), str(e))

    # M1: _collect_images center-crops mixed-resolution batches to min (H, W).
    if torch is None:
        print("[SKIP] torch not available — _collect_images mixed-resolution checks skipped")
    else:
        big = torch.zeros(1, 64, 96, 3)
        big[0, 8:56, 24:72, :] = 1.0  # centered 48x48 patch
        small = torch.full((1, 48, 48, 3), 2.0)
        batched = nodes._collect_images({"image_0": big, "image_1": small})
        check("_collect_images crops to min (H, W)",
              tuple(batched.shape) == (2, 48, 48, 3), f"got {tuple(batched.shape)}")
        check("_collect_images center-crops (equal both sides)",
              bool((batched[0] == 1.0).all()) and bool((batched[1] == 2.0).all()))
        odd = torch.zeros(1, 5, 7, 3)
        odd[0, 1:4, 2:5, :] = 3.0  # centered 3x3 patch (odd dims: extra pixel bottom/right)
        even = torch.full((1, 3, 3, 3), 4.0)
        b2 = nodes._collect_images([odd, even])
        check("_collect_images handles odd crop offsets",
              tuple(b2.shape) == (2, 3, 3, 3) and bool((b2[0] == 3.0).all()))
        single = torch.zeros(1, 10, 12, 3)
        check("_collect_images passes single image through untouched",
              nodes._collect_images({"image_0": single}) is single)
        check("_collect_images returns None for empty payload",
              nodes._collect_images(None) is None and nodes._collect_images({}) is None)

    # analyzer execute path with dict autogrow payload (needs torch -> skip image tensor;
    # single image object works without torch)
    analyzer_out = nodes.H3VisionAnalyzer.execute(
        text_encoder="fake_qwen_vl.safetensors", vision_mode="detailed_subject_scene",
        seed=1, temperature=0.2, max_tokens=256, keep_model_loaded=True,
        images={"image_0": FakeImage()},
    )
    check("analyzer execute returns NodeOutput with description",
          isinstance(analyzer_out, _NodeOutput) and len(analyzer_out.args) == 1
          and isinstance(analyzer_out.args[0], str) and analyzer_out.args[0])

    # ------------------------------------------------------------------
    # 11. Planner mode end-to-end on a scripted fake CLIP
    # ------------------------------------------------------------------
    PLANNER_ANSWER = (
        "=== GLOBAL ===\n"
        "A pale woman in a dark ceremonial robe inside an ancient temple. Cold metallic\n"
        "architecture, dim amber ritual light.\n"
        "\n"
        "=== CLIPS ===\n"
        "clip_1:\n"
        "The woman walks toward the central altar. Her robe moves naturally with each step.\n"
        "\n"
        "clip_2:\n"
        "She continues the same walk and gradually raises her right hand.\n"
        "\n"
        "clip_3:\n"
        "Her raised hand reaches the altar surface and the chamber fills with warm light.\n"
        "\n"
        "=== CAMERAS ===\n"
        '[{"clip_id": "clip-1", "clip_name": "Approach", "shot_size": "Medium Shot", '
        '"rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Natural 35mm", '
        '"stabilization": "Gimbal Smooth", "movement": "Track Forward", "speed": "Slow", '
        '"transition_type": "Continuous / Same Shot", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": true}, '
        '{"clip_id": "clip-2", "clip_name": "Raise", "shot_size": "Medium Close-Up", '
        '"rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Portrait 65mm", '
        '"stabilization": "Gimbal Smooth", "movement": "Push-In", "speed": "Slow", '
        '"transition_type": "Continuous / Same Shot", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": true}, '
        '{"clip_id": "clip-3", "clip_name": "Touch", "shot_size": "Close-Up", '
        '"rig": "Tripod / Locked Head", "camera_body": "Cinematic Neutral", '
        '"lens": "Portrait 85mm", "stabilization": "Hard Locked", '
        '"movement": "Locked-Off / Static", "speed": "Static", '
        '"transition_type": "Continuous / Same Shot", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": false}]'
    )
    RU_ANSWER = (
        "GLOBAL:\n"
        "Бледная женщина в тёмном церемониальном одеянии внутри древнего храма.\n"
        "\n"
        "clip_1:\n"
        "Женщина идёт к центральному алтарю. Одеяние естественно движется при каждом шаге.\n"
        "\n"
        "clip_2:\n"
        "Она продолжает тот же путь и постепенно поднимает правую руку.\n"
        "\n"
        "clip_3:\n"
        "Поднятая рука касается поверхности алтаря, и зал наполняется тёплым светом."
    )

    class FakePlannerClip:
        """Generative CLIP fake scripted for the two planner-mode passes."""

        def __init__(self):
            self.calls = []

        def tokenize(self, prompt, image=None, skip_template=False, min_length=1):
            self.calls.append(("tokenize", len(prompt)))
            return {"prompt": prompt, "image": image}

        def generate(self, tokens, do_sample=False, max_length=512, temperature=1.0,
                     top_k=50, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
                     presence_penalty=0.0, seed=None):
            self.calls.append(("generate", max_length, temperature, seed))
            return [1, 2, 3]

        def decode(self, ids):
            n_generate = sum(1 for c in self.calls if c[0] == "generate")
            return PLANNER_ANSWER if n_generate == 1 else RU_ANSWER

    fpc = FakePlannerClip()
    planner_out = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
        duration=8.0, vision_mode="skip (idea only)", seed=7, temperature=0.7,
        top_p=0.95, top_k=64, max_tokens=256, variants=3,
        use_default_template=False, keep_model_loaded=True,
        clip=fpc, images=None, extra_instructions="", custom_system_prompt="",
        emit_planner_outputs=True, total_duration=24.0,
    )
    args = planner_out.args if isinstance(planner_out, _NodeOutput) else planner_out.result
    check("planner: NodeOutput has 9 values", len(args) == 9, f"got {len(args)}")
    check("planner: legacy sockets empty", args[0] == "" and args[1] == "")
    check("planner: planner_prompt has 3 clip sections",
          args[3].count("clip_") == 3 and args[3].startswith("clip_1:"))
    check("planner: planner_prompt_ru mirror present",
          args[4].startswith("GLOBAL:") and "clip_1:" in args[4] and "алтарю" in args[4])
    check("planner: global_prompt extracted",
          "ceremonial robe" in args[5] and "===" not in args[5])
    check("planner: clip_durations JSON",
          json.loads(args[6]) == [8.0, 8.0, 8.0], f"got {args[6]}")
    cards = json.loads(args[7])
    check("planner: camera_params 3 validated cards",
          len(cards) == 3 and cards[0]["shot_size"] == "Medium Shot"
          and cards[-1]["transition_to_next"] is False
          and list(cards[0].keys()) == list(nodes.camera_knowledge.CAMERA_CARD_KEYS))
    check("planner: reference_images_status none",
          json.loads(args[8]) == {"images_connected": False, "count": 0, "slots": []})
    pdebug = json.loads(args[2])
    check("planner: debug mode + variants forced warning",
          pdebug["mode"] == "planner"
          and any("variants=3 ignored" in w for w in pdebug["warnings"]))
    check("planner: debug carries plan fields",
          pdebug["clip_count"] == 3 and pdebug["clips_parsed"] == 3
          and pdebug["cameras_validated"] == 3 and pdebug["total_duration"] == 24.0)
    check("planner: clip_duration=0 falls back to duration widget",
          pdebug["target_clip_duration"] == 8.0
          and pdebug["target_clip_duration_source"] == "duration")
    gen_calls = [c for c in fpc.calls if c[0] == "generate"]
    check("planner: exactly two generation passes", len(gen_calls) == 2,
          f"got {len(gen_calls)}")
    check("planner: planner pass budget independent of max_tokens widget",
          gen_calls[0][1] >= 2048 and gen_calls[0][1] != 256, f"got {gen_calls[0][1]}")
    check("planner: translate pass uses fidelity temperature",
          abs(gen_calls[1][2] - 0.3) < 1e-9 and gen_calls[1][1] >= 1024)

    # 11b. Planner mode with clip_duration > 15s (long clips beyond the
    # original node's single-video cap): 60s total at a 30s clip target
    # must yield exactly 2 clips of 30s each.
    PLANNER_ANSWER_2 = (
        "=== GLOBAL ===\n"
        "A pale woman in a dark ceremonial robe inside an ancient temple.\n"
        "\n"
        "=== CLIPS ===\n"
        "clip_1:\n"
        "The woman walks toward the central altar across the long ceremonial hall.\n"
        "\n"
        "clip_2:\n"
        "She reaches the altar and the chamber fills with warm light.\n"
        "\n"
        "=== CAMERAS ===\n"
        '[{"clip_id": "clip-1", "clip_name": "Approach", "shot_size": "Medium Shot", '
        '"rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Natural 35mm", '
        '"stabilization": "Gimbal Smooth", "movement": "Track Forward", "speed": "Slow", '
        '"transition_type": "Continuous / Same Shot", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": true}, '
        '{"clip_id": "clip-2", "clip_name": "Arrival", "shot_size": "Wide Shot", '
        '"rig": "Tripod / Locked Head", "camera_body": "Cinematic Neutral", '
        '"lens": "Natural 24mm", "stabilization": "Hard Locked", '
        '"movement": "Locked-Off / Static", "speed": "Static", '
        '"transition_type": "Cut", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": false}]'
    )
    RU_ANSWER_2 = (
        "GLOBAL:\n"
        "Бледная женщина в тёмном церемониальном одеянии внутри древнего храма.\n"
        "\n"
        "clip_1:\n"
        "Женщина идёт к центральному алтарю через длинный церемониальный зал.\n"
        "\n"
        "clip_2:\n"
        "Она достигает алтаря, и зал наполняется тёплым светом."
    )

    class FakePlannerClip2(FakePlannerClip):
        """Qwen-family fake: a tokenizer whose class name sniffs as 'qwen'
        (engine.detect_family), so the planner pass goes through the qwen chat
        template + the think suppressor — mirroring the user's Qwen3-VL setup."""

        def __init__(self):
            super().__init__()
            self.prompts = []
            self.tokenizer = type("Qwen3VLTokenizer", (), {})()

        def tokenize(self, prompt, image=None, skip_template=False, min_length=1):
            self.prompts.append(str(prompt))
            return super().tokenize(prompt, image, skip_template, min_length)

        def decode(self, ids):
            n_generate = sum(1 for c in self.calls if c[0] == "generate")
            return PLANNER_ANSWER_2 if n_generate == 1 else RU_ANSWER_2

    fpc2 = FakePlannerClip2()
    planner_out2 = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
        duration=8.0, vision_mode="skip (idea only)", seed=7, temperature=0.7,
        top_p=0.95, top_k=64, max_tokens=256, variants=1,
        use_default_template=False, keep_model_loaded=True,
        clip=fpc2, images=None, extra_instructions="", custom_system_prompt="",
        emit_planner_outputs=True, total_duration=60.0, clip_duration=30.0,
    )
    args2 = planner_out2.args if isinstance(planner_out2, _NodeOutput) else planner_out2.result
    check("planner long clips: 2 clips of 30s",
          json.loads(args2[6]) == [30.0, 30.0], f"got {args2[6]}")
    check("planner long clips: planner_prompt has 2 sections",
          args2[3].count("clip_") == 2 and args2[3].startswith("clip_1:"))
    check("planner long clips: 2 camera cards",
          len(json.loads(args2[7])) == 2)
    pdebug2 = json.loads(args2[2])
    check("planner long clips: clip_duration overrides duration",
          pdebug2["target_clip_duration"] == 30.0
          and pdebug2["target_clip_duration_source"] == "clip_duration"
          and pdebug2["clip_count"] == 2 and pdebug2["total_duration"] == 60.0)
    check("planner long clips: qwen family flags in debug (v1.1.4)",
          pdebug2["model_family"] == "qwen"
          and pdebug2["thinking_suppressed"] is True
          and pdebug2["chat_format"] == "chatml"
          and "qwen" in pdebug2["family_signals"].lower())
    gen2 = [c for c in fpc2.calls if c[0] == "generate"]
    check("planner long clips: exactly two generation passes", len(gen2) == 2,
          f"got {len(gen2)}")
    planner_user_msg = fpc2.prompts[0] if fpc2.prompts else ""
    check("planner long clips: prompt carries 60s/2x30s constraint",
          "60 seconds in total" in planner_user_msg and "30s" in planner_user_msg
          and "2 consecutive clips" in planner_user_msg)
    budget30 = nodes.planner_mode.per_clip_word_budget(30.0)
    check("planner long clips: prompt carries per-clip word budgets",
          f"clip_1 ~{budget30} words" in planner_user_msg
          and f"clip_2 ~{budget30} words" in planner_user_msg,
          f"budget30={budget30}")
    check("planner long clips: token budget scaled with durations",
          gen2[0][1] >= 2048, f"got {gen2[0][1]}")
    check("planner long clips: planner pass prompt ends with the Qwen3 empty think block",
          planner_user_msg.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
          and planner_user_msg.startswith("<|im_start|>system\n"),
          "suppress_thinking must append the Qwen3 empty-think-block after the assistant marker")

    # ------------------------------------------------------------------
    # 11b-boost. Creative boost defaults (v1.1.5): the 11b run above used the
    # widget defaults, i.e. creative_boost='standard' — the planner system
    # prompt must carry the CREATIVE WRITING STANDARDS block and the RU
    # translate pass must carry the artistic-accuracy note, out of the box.
    # ------------------------------------------------------------------
    check("creative default: level 'standard' in debug",
          pdebug2["creative_boost"] == "standard"
          and pdebug2["style_directive"] is None
          and pdebug2["motif_hints"] is None)
    check("creative default: planner system carries the writing standards",
          "CREATIVE WRITING STANDARDS" in planner_user_msg
          and "EXPRESSIVE MODE" not in planner_user_msg)
    check("creative default: translate system carries the vividness note",
          "ХУДОЖЕСТВЕННАЯ ТОЧНОСТЬ" in fpc2.prompts[1])

    # ------------------------------------------------------------------
    # 11f. Creative boost 'expressive' + style_directive (v1.1.5): the user
    # message must carry the author's brief and the sampled per-clip motif
    # ingredients; the system prompt must add the EXPRESSIVE addendum; the
    # plan itself must still parse exactly like before (creativity must
    # never break the output contract).
    # ------------------------------------------------------------------
    class FakePlannerClip3(FakePlannerClip2):
        pass

    fpc_boost = FakePlannerClip3()
    planner_boost = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
        duration=8.0, vision_mode="skip (idea only)", seed=42, temperature=0.7,
        top_p=0.95, top_k=64, max_tokens=256, variants=1,
        use_default_template=False, keep_model_loaded=True,
        clip=fpc_boost, images=None, extra_instructions="", custom_system_prompt="",
        emit_planner_outputs=True, total_duration=60.0, clip_duration=30.0,
        creative_boost="expressive",
        style_directive="rain-soaked neon noir, melancholy, Wong Kar-wai colours",
    )
    bargs = planner_boost.args if isinstance(planner_boost, _NodeOutput) else planner_boost.result
    bdebug = json.loads(bargs[2])
    boost_prompt = fpc_boost.prompts[0]
    check("creative expressive: plan still parses (2 clips of 30s)",
          json.loads(bargs[6]) == [30.0, 30.0] and bargs[3].count("clip_") == 2
          and bargs[4].startswith("GLOBAL:"))
    check("creative expressive: system prompt carries standards + addendum",
          "CREATIVE WRITING STANDARDS" in boost_prompt and "EXPRESSIVE MODE" in boost_prompt)
    check("creative expressive: user message carries the author's brief",
          "Creative direction" in boost_prompt
          and "rain-soaked neon noir, melancholy, Wong Kar-wai colours" in boost_prompt)
    check("creative expressive: motif ingredients in the user message",
          "Suggested visual motifs" in boost_prompt and "world spine" in boost_prompt
          and "clip_1 beat" in boost_prompt and "clip_2 beat" in boost_prompt)
    check("creative expressive: translate pass keeps the vividness note",
          "ХУДОЖЕСТВЕННАЯ ТОЧНОСТЬ" in fpc_boost.prompts[1])
    check("creative expressive: debug echoes level, brief and hints",
          bdebug["creative_boost"] == "expressive"
          and "neon noir" in (bdebug["style_directive"] or "")
          and isinstance(bdebug["motif_hints"], dict)
          and set(bdebug["motif_hints"].keys()) == {"world", "clips"}
          and len(bdebug["motif_hints"]["world"]) == 2
          and set(bdebug["motif_hints"]["clips"].keys()) == {"1", "2"}
          and all(len(v) == 1 for v in bdebug["motif_hints"]["clips"].values()))
    # Determinism: same seed -> same motif ingredients across runs.
    hints_same = nodes.planner_mode.sample_motif_hints(2, 42)
    check("creative expressive: motif hints deterministic in the seed",
          tuple(bdebug["motif_hints"]["world"]) == tuple(hints_same["world"])
          and tuple(bdebug["motif_hints"]["clips"]["1"]) == tuple(hints_same["clips"][1])
          and tuple(bdebug["motif_hints"]["clips"]["2"]) == tuple(hints_same["clips"][2]))

    # ------------------------------------------------------------------
    # 11g. Creative boost 'off' (v1.1.5 regression): byte-exact v1.1.4
    # prompts — no standards block, no motif ingredients, no translate note,
    # while the plan still parses.
    # ------------------------------------------------------------------
    class FakePlannerClip4(FakePlannerClip2):
        pass

    fpc_off = FakePlannerClip4()
    planner_off = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
        duration=8.0, vision_mode="skip (idea only)", seed=42, temperature=0.7,
        top_p=0.95, top_k=64, max_tokens=256, variants=1,
        use_default_template=False, keep_model_loaded=True,
        clip=fpc_off, images=None, extra_instructions="", custom_system_prompt="",
        emit_planner_outputs=True, total_duration=60.0, clip_duration=30.0,
        creative_boost="off",
    )
    oargs = planner_off.args if isinstance(planner_off, _NodeOutput) else planner_off.result
    odebug = json.loads(oargs[2])
    off_prompt = fpc_off.prompts[0]
    check("creative off: plan still parses",
          json.loads(oargs[6]) == [30.0, 30.0] and oargs[3].count("clip_") == 2)
    check("creative off: no standards, no motifs, no brief anywhere",
          "CREATIVE WRITING STANDARDS" not in off_prompt
          and "EXPRESSIVE MODE" not in off_prompt
          and "Suggested visual motifs" not in off_prompt
          and "Creative direction" not in off_prompt)
    check("creative off: translate prompt without the vividness note",
          "ХУДОЖЕСТВЕННАЯ ТОЧНОСТЬ" not in fpc_off.prompts[1])
    check("creative off: debug echoes the off level",
          odebug["creative_boost"] == "off" and odebug["motif_hints"] is None
          and odebug["style_directive"] is None)
    # The 'off' planner chat prompt must equal the 11b 'standard' prompt with
    # ONLY the standards block removed (byte-exact regression guard).
    std_block = nodes.planner_mode.CREATIVE_STANDARDS_TEXT
    check("creative off: prompt == standard prompt minus the standards block",
          off_prompt == planner_user_msg.replace("\n\n" + std_block, ""))

    # ------------------------------------------------------------------
    # 11h. Lost GLOBAL guard (v1.1.6): when the model answer has parseable
    # clips but NO GLOBAL section, the node must warn loudly (LongMedia joins
    # global_prompt with every clip at runtime — an empty global turns the
    # sequence into independent scenes), keep global_prompt empty and report
    # global_prompt_chars=0 in the debug output.
    # ------------------------------------------------------------------
    PLANNER_ANSWER_NOGLOBAL = (
        "=== CLIPS ===\n"
        "clip_1:\n"
        "The woman walks toward the central altar across the long ceremonial hall.\n"
        "\n"
        "clip_2:\n"
        "She reaches the altar and the chamber fills with warm light.\n"
        "\n"
        "=== CAMERAS ===\n"
        '[{"clip_id": "clip-1", "clip_name": "Approach", "shot_size": "Medium Shot", '
        '"rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Natural 35mm", '
        '"stabilization": "Gimbal Smooth", "movement": "Track Forward", "speed": "Slow", '
        '"transition_type": "Continuous / Same Shot", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": true}, '
        '{"clip_id": "clip-2", "clip_name": "Arrival", "shot_size": "Wide Shot", '
        '"rig": "Tripod / Locked Head", "camera_body": "Cinematic Neutral", '
        '"lens": "Natural 35mm", "stabilization": "Gimbal Smooth", "movement": "Static", '
        '"speed": "Slow", "transition_type": "Cut", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": false}]'
    )

    class FakePlannerClip5(FakePlannerClip2):
        def decode(self, ids):
            n_generate = sum(1 for c in self.calls if c[0] == "generate")
            return PLANNER_ANSWER_NOGLOBAL if n_generate == 1 else RU_ANSWER_2

    fpc_nog = FakePlannerClip5()
    planner_nog = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
        duration=8.0, vision_mode="skip (idea only)", seed=42, temperature=0.7,
        top_p=0.95, top_k=64, max_tokens=256, variants=1,
        use_default_template=False, keep_model_loaded=True,
        clip=fpc_nog, images=None, extra_instructions="", custom_system_prompt="",
        emit_planner_outputs=True, total_duration=60.0, clip_duration=30.0,
        creative_boost="off",
    )
    nargs = planner_nog.args if isinstance(planner_nog, _NodeOutput) else planner_nog.result
    ndebug = json.loads(nargs[2])
    check("lost global: clips still parse and planner_prompt survives",
          nargs[3].count("clip_") == 2 and json.loads(nargs[6]) == [30.0, 30.0])
    check("lost global: global_prompt output stays empty",
          nargs[5] == "" and ndebug["global_prompt_chars"] == 0)
    check("lost global: loud warning names the consequence and the wiring",
          any("GLOBAL section is missing or empty" in w
              and "independent scenes" in w and "global_prompt" in w
              for w in ndebug["warnings"]))

    # ------------------------------------------------------------------
    # 11i. Reference-image photo-meta leak (v1.1.7): with a ref image
    # connected, the vision dump must NOT be copied into GLOBAL ("The photo
    # shows ..." -> RU mirror "На фото изображено ... и полный вижн рефа").
    # The vision pass system prompt bans photo-artifact wording; the planner
    # user message frames the analysis as RAW MATERIAL with the <image_0> ->
    # <Picture 1> mapping; the deterministic scrubber removes any leaked
    # sentences from the final EN outputs and the RU mirror.
    # ------------------------------------------------------------------
    VISION_ANSWER_LEAK = (
        "A young woman in her late twenties with shoulder-length dark wavy hair, "
        "wearing a red waterproof jacket, standing in a kitchen."
    )
    PLANNER_ANSWER_LEAK = (
        "=== GLOBAL ===\n"
        "The photo shows a young woman in her late twenties with shoulder-length "
        "dark wavy hair, wearing a red waterproof jacket, posing in a kitchen. "
        "The video follows this courier through a rain-soaked harbour city at dusk.\n"
        "\n"
        "=== CLIPS ===\n"
        "clip_1:\n"
        "The courier steps off the ferry into the rain, the same action continues.\n"
        "\n"
        "clip_2:\n"
        "As shown in the reference image, she keeps her hood up. "
        "She continues walking through the fish market.\n"
        "\n"
        "=== CAMERAS ===\n"
        '[{"clip_id": "clip-1", "clip_name": "Ferry", "shot_size": "Medium Shot", '
        '"rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Natural 35mm", '
        '"stabilization": "Gimbal Smooth", "movement": "Track Forward", "speed": "Slow", '
        '"transition_type": "Continuous / Same Shot", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": true}, '
        '{"clip_id": "clip-2", "clip_name": "Market", "shot_size": "Wide Shot", '
        '"rig": "Tripod / Locked Head", "camera_body": "Cinematic Neutral", '
        '"lens": "Natural 24mm", "stabilization": "Hard Locked", '
        '"movement": "Locked-Off / Static", "speed": "Static", '
        '"transition_type": "Cut", "space_relation": "Same Space", '
        '"entity_continuity": "Lock Population / Layout", "transition_to_next": false}]'
    )
    RU_ANSWER_LEAK = (
        "GLOBAL:\n"
        "На фото изображена молодая женщина в красной куртке, она курьер в "
        "дождливом портовом городе. Видео следует за ней.\n"
        "\n"
        "clip_1:\n"
        "Курьер сходит с парома под дождём.\n"
        "\n"
        "clip_2:\n"
        "Она продолжает идти через рыбный рынок."
    )

    class FakePlannerClipRef(FakePlannerClip2):
        """Three scripted passes: vision -> planner -> translate."""

        def decode(self, ids):
            n_generate = sum(1 for c in self.calls if c[0] == "generate")
            if n_generate == 1:
                return VISION_ANSWER_LEAK
            if n_generate == 2:
                return PLANNER_ANSWER_LEAK
            return RU_ANSWER_LEAK

    fpc_ref = FakePlannerClipRef()
    planner_ref = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="take the face and hairstyle from <image_0>",
        task_type="Auto", duration=8.0, vision_mode="detailed_subject_scene", seed=7,
        temperature=0.7, top_p=0.95, top_k=64, max_tokens=256, variants=1,
        use_default_template=False, keep_model_loaded=True,
        clip=fpc_ref, images={"image_0": FakeImage()}, extra_instructions="",
        custom_system_prompt="", emit_planner_outputs=True,
        total_duration=60.0, clip_duration=30.0,
    )
    rargs = planner_ref.args if isinstance(planner_ref, _NodeOutput) else planner_ref.result
    rdebug = json.loads(rargs[2])
    check("ref leak: vision pass uses the photo-artifact-free system prompt",
          "never refer to the photograph" in fpc_ref.prompts[0]
          and "no framing, shot type" in fpc_ref.prompts[0])
    planner_ref_msg = fpc_ref.prompts[1]
    check("ref leak: planner user message frames the analysis as RAW MATERIAL",
          "RAW MATERIAL" in planner_ref_msg
          and "<image_0> == <Picture 1>" in planner_ref_msg
          and VISION_ANSWER_LEAK[:40] in planner_ref_msg
          and "Notation mapping" in planner_ref_msg)
    check("ref leak: GLOBAL carries video constants, not the photo dump",
          "The photo shows" not in rargs[5] and "kitchen" not in rargs[5]
          and "posing" not in rargs[5] and "courier" in rargs[5])
    check("ref leak: clips scrubbed of source-image references",
          "reference image" not in rargs[3].lower()
          and rargs[3].count("clip_") == 2
          and "fish market" in rargs[3])
    check("ref leak: RU mirror scrubbed of the photo opener",
          "на фото" not in rargs[4].lower() and rargs[4].startswith("GLOBAL:")
          and "clip_1:" in rargs[4])
    check("ref leak: scrub warnings + debug counter",
          rdebug["photo_meta_scrubbed"] >= 3
          and any("photo-meta" in w for w in rdebug["warnings"]))
    check("ref leak: reference_images_status reports the connected slot",
          json.loads(rargs[8]) == {"images_connected": True, "count": 1,
                                   "slots": ["image_0"]})
    check("ref leak: plan shape survives (2 clips, 2 cards, durations)",
          json.loads(rargs[6]) == [30.0, 30.0] and len(json.loads(rargs[7])) == 2)

    # ------------------------------------------------------------------
    # 11c. Degenerate planner answer (the user-reported "пустышка"): the
    # model burns its budget inside a think block full of dots. The planner
    # text outputs must be empty BUT debug must now carry the raw answer
    # excerpt + an explicit warning instead of failing silently.
    # ------------------------------------------------------------------
    DEGENERATE_ANSWER = (
        "<think>\n"
        + ("................................................................\n" * 24)
        + "</think>"
    )

    class FakeDegenerateClip(FakePlannerClip):
        def decode(self, ids):
            n_generate = sum(1 for c in self.calls if c[0] == "generate")
            return DEGENERATE_ANSWER if n_generate == 1 else ""

    fpc3 = FakeDegenerateClip()
    planner_out3 = nodes.H3VisionPromptor.execute(
        text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
        duration=10.0, vision_mode="skip (idea only)", seed=7, temperature=0.4,
        top_p=0.95, top_k=64, max_tokens=2048, variants=1,
        use_default_template=True, keep_model_loaded=True,
        clip=fpc3, images=None, extra_instructions="", custom_system_prompt="",
        emit_planner_outputs=True, total_duration=60.0, clip_duration=10.0,
    )
    args3 = planner_out3.args if isinstance(planner_out3, _NodeOutput) else planner_out3.result
    check("degenerate: planner text outputs empty",
          args3[3] == "" and args3[4] == "" and args3[5] == "")
    check("degenerate: clip_durations still computed",
          json.loads(args3[6]) == [10.0, 10.0, 10.0, 10.0, 10.0, 10.0], f"got {args3[6]}")
    check("degenerate: camera_params empty array", args3[7] == "[]")
    ddebug = json.loads(args3[2])
    check("degenerate: debug carries raw answer excerpt",
          ddebug["raw_planner_answer_chars"] == len(DEGENERATE_ANSWER)
          and ddebug["raw_planner_answer"].startswith("<think>\n")
          and "...." in ddebug["raw_planner_answer"],
          "raw_planner_answer must show the un-cleaned model output")
    check("degenerate: debug flags the parse failure",
          any("no parseable clip sections" in w for w in ddebug["warnings"]))
    check("degenerate: debug records template + honest family state (v1.1.4)",
          ddebug["use_default_template"] is True
          and ddebug["model_family"] == "generic"
          and ddebug["thinking_suppressed"] is False
          and ddebug["chat_format"] == "raw (no chat template)")
    check("degenerate: generic-family no-template warning present",
          any("model_family='generic'" in w and "RAW text" in w
              for w in ddebug["warnings"]))
    check("degenerate: no translate pass (no clips)",
          ddebug["translate_token_budget"] == 0
          and ddebug["raw_translate_answer"] is None)

    # ------------------------------------------------------------------
    # 11d. Engine template unit checks: the think suppressor is qwen-only
    # and opt-in (legacy default unchanged).
    # ------------------------------------------------------------------
    # engine is already imported from the fork package at the top of main().
    h3_engine = engine
    q_base = h3_engine.build_chat_text("qwen", "SYS", "USR", 0, None)
    q_supp = h3_engine.build_chat_text("qwen", "SYS", "USR", 0, None, suppress_thinking=True)
    check("engine: default qwen template has NO think block (legacy identical)",
          q_base.endswith("<|im_start|>assistant\n") and "<think>" not in q_base)
    check("engine: suppressed qwen template ends with the empty think block",
          q_supp.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    g_supp = h3_engine.build_chat_text("gemma", "SYS", "USR", 0, None, suppress_thinking=True)
    check("engine: gemma template unaffected by suppress_thinking",
          "<think>" not in g_supp and g_supp.startswith("<start_of_turn>user\n"))
    q_img = h3_engine.build_chat_text("qwen", "SYS", "USR", 2, None, suppress_thinking=True)
    check("engine: vision block preserved with suppression",
          q_img.count("<|image_pad|>") == 2 and q_img.endswith("<think>\n\n</think>\n\n"))

    # ------------------------------------------------------------------
    # 11e. MiniMax-H3 conditioning encoder rejection (v1.1.4): the H3 text
    # encoder file (qwen3vl_32b_minimax_h3_*) routes to MiniMaxH3Tokenizer,
    # which never chat-templates (and the checkpoint is truncated to 50
    # layers with no final norm). The fork must fail FAST in BOTH modes with
    # an actionable error instead of generating punctuation garbage — this
    # is exactly the user-reported "2048 commas" scenario.
    # ------------------------------------------------------------------
    class _MiniMaxQwen3VL:
        pass

    class _MiniMaxH3TEModel:
        def __init__(self):
            self.clip_name = "qwen3vl_32b"
            self.qwen3vl_32b = types.SimpleNamespace(transformer=_MiniMaxQwen3VL())

    class _MiniMaxH3Tokenizer:
        def __init__(self):
            self.clip_name = "qwen3vl_32b"
            self.qwen3vl_32b = type("MiniMaxQwenSDTokenizer", (), {})()

    class FakeH3EncoderClip(FakePlannerClip):
        """Mimics comfy.sd.CLIP for the MiniMax-H3 conditioning encoder
        (comfy.text_encoders.minimax.MiniMaxH3TEModel + MiniMaxH3Tokenizer).
        The nested-key layout mirrors SD1ClipModel/SD1Tokenizer, where the
        real objects hide under attributes named after clip_name."""

        def __init__(self):
            super().__init__()
            self.cond_stage_model = _MiniMaxH3TEModel()
            self.tokenizer = _MiniMaxH3Tokenizer()

    fh3 = FakeH3EncoderClip()
    fam_h3 = engine.detect_family(fh3)
    check("h3 encoder: detect_family prefers 'minimax_h3' over 'qwen'",
          fam_h3 == "minimax_h3",
          f"got {fam_h3!r} — signals contain both 'minimax' and 'qwen' (qwen3vl_32b key)")
    check("h3 encoder: nested-key signals collected (clip_name key + inner classes)",
          "qwen3vl_32b" in engine.family_signals_text(fh3)
          and "minimaxh3tokenizer" in engine.family_signals_text(fh3).lower())

    raised_planner = None
    try:
        nodes.H3VisionPromptor.execute(
            text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
            duration=10.0, vision_mode="skip (idea only)", seed=7, temperature=0.4,
            top_p=0.95, top_k=64, max_tokens=2048, variants=1,
            use_default_template=True, keep_model_loaded=True,
            clip=fh3, images=None, extra_instructions="", custom_system_prompt="",
            emit_planner_outputs=True, total_duration=60.0, clip_duration=10.0,
        )
    except RuntimeError as e:
        raised_planner = str(e)
    check("h3 encoder: planner mode fails fast with the actionable error",
          raised_planner is not None and "MiniMax" in raised_planner
          and "instruct" in raised_planner and "MiniMaxH3Tokenizer" in raised_planner,
          f"raised={raised_planner!r}")
    check("h3 encoder: planner fail is instant (no generation burned)",
          not any(c[0] == "generate" for c in fh3.calls),
          "assert_chat_capable must raise before any generate() call")

    raised_legacy = None
    try:
        nodes.H3VisionPromptor.execute(
            text_encoder="ignored", user_idea="a ritual in a temple", task_type="Auto",
            duration=10.0, vision_mode="skip (idea only)", seed=7, temperature=0.4,
            top_p=0.95, top_k=64, max_tokens=2048, variants=1,
            use_default_template=True, keep_model_loaded=True,
            clip=fh3, images=None, extra_instructions="", custom_system_prompt="",
        )
    except RuntimeError as e:
        raised_legacy = str(e)
    check("h3 encoder: original mode also refuses the conditioning encoder",
          raised_legacy is not None and "MiniMax" in raised_legacy,
          f"raised={raised_legacy!r}")

    # ------------------------------------------------------------------
    # 12. Original mode behavior-identical to the upstream package
    # ------------------------------------------------------------------
    orig_pkg_dir = PKG_DIR.parent / "ComfyUI-H3-VisionPromptor"
    if not orig_pkg_dir.is_dir():
        print("[SKIP] upstream package not found next to the fork — identity check skipped")
    else:
        spec2 = importlib.util.spec_from_file_location(
            "h3_vision_promptor_orig", orig_pkg_dir / "__init__.py",
            submodule_search_locations=[str(orig_pkg_dir)],
        )
        orig_pkg = importlib.util.module_from_spec(spec2)
        sys.modules[spec2.name] = orig_pkg
        spec2.loader.exec_module(orig_pkg)
        from h3_vision_promptor_orig.py import nodes as orig_nodes

        def _run_legacy(node_cls, clip_obj):
            return node_cls.execute(
                text_encoder="ignored", user_idea="a cat naps on a sofa",
                task_type="Auto", duration=8.0, vision_mode="skip (idea only)",
                seed=42, temperature=0.7, top_p=0.95, top_k=64, max_tokens=512,
                variants=1, use_default_template=False, keep_model_loaded=True,
                clip=clip_obj, images=None, extra_instructions="keep it calm",
                custom_system_prompt="",
            )

        out_orig = _run_legacy(orig_nodes.H3VisionPromptor, FakeClip())
        out_fork = _run_legacy(nodes.H3VisionPromptor, FakeClip())
        a_orig = out_orig.args if isinstance(out_orig, _NodeOutput) else out_orig.result
        a_fork = out_fork.args if isinstance(out_fork, _NodeOutput) else out_fork.result
        check("legacy: prompt identical to upstream", a_fork[0] == a_orig[0])
        check("legacy: vision_context identical to upstream", a_fork[1] == a_orig[1])
        d_orig, d_fork = json.loads(a_orig[2]), json.loads(a_fork[2])
        for k in ("vision_seconds", "generation_seconds", "total_seconds"):
            d_orig.pop(k, None), d_fork.pop(k, None)
        check("legacy: debug identical to upstream (modulo timings)", d_orig == d_fork)
        check("legacy: planner sockets empty", a_fork[3] == "" and a_fork[8] == "")

    print()
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("SMOKE TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
