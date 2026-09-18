#!/usr/bin/env python3
"""Unit tests for the fork's planner pipeline (py/planner_mode.py,
py/camera_knowledge.py, py/prompt_builder.py planner builders).

Runs OUTSIDE ComfyUI — these modules import no ComfyUI code.
A faithful re-implementation of LongMedia's multiclip import parser
(_V046_MULTICLIP_HEADER_RE + header-section walk) cross-checks that every
assembled planner_prompt is importable by the real LongMedia Planner.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _load_module(relpath, name):
    spec = importlib.util.spec_from_file_location(name, PKG_DIR / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


camera_knowledge = _load_module("py/camera_knowledge.py", "h3vp_camera_knowledge")
# planner_mode does `from . import camera_knowledge` — give it a package parent.
import types

pkg = types.ModuleType("h3vp_pkg")
pkg.__path__ = [str(PKG_DIR)]
pkg.camera_knowledge = camera_knowledge
sys.modules["h3vp_pkg"] = pkg
sys.modules["h3vp_pkg.camera_knowledge"] = camera_knowledge
planner_mode = _load_module("py/planner_mode.py", "h3vp_pkg.planner_mode")
prompt_builder = _load_module("py/prompt_builder.py", "h3vp_pkg.prompt_builder")


# ---------------------------------------------------------------------------
# LongMedia import-parser twin (nodes.py @ vizart-vj/ComfyUI-MiniMax-H3-LongMedia)
# ---------------------------------------------------------------------------

LM_HEADER_RE = re.compile(
    r"^\[?(?:clip|shot)[ _-]*(\d{1,2})\]?(?:\s*\([^)]*\))?\s*(?:(?::|=|[-–—])\s*)?(.*)$",
    re.IGNORECASE,
)


def lm_header(line):
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", str(line or "")).strip()
    s = re.sub(r"^\s*(?:[-+]\s+)(?=(?:\*\*)?\[?(?:clip|shot)\b)", "", s, flags=re.I)
    s = s.replace("**", "").replace("`", "").strip()
    m = LM_HEADER_RE.match(s)
    if not m:
        return None
    idx = int(m.group(1))
    if idx < 1 or idx > 16:
        raise ValueError("index out of range")
    return idx, str(m.group(2) or "").strip()


def lm_import_sections(text):
    """Mirror of LongMedia's _v046_try_header_sections (no YAML unwrapping —
    plain bodies; enough to prove importability of our output)."""
    sections = {}
    current = None
    body = []

    def flush():
        nonlocal current, body
        if current is None or current in sections:
            return
        sections[current] = "\n".join(body).strip()
        body = []

    for line in str(text).split("\n"):
        header = lm_header(line)
        if header is not None:
            flush()
            current, inline = header
            body = [inline] if inline else []
        elif current is not None:
            body.append(line)
    flush()
    if len(sections) < 2:
        return None
    max_idx = max(sections)
    if sorted(sections) != list(range(1, max_idx + 1)):
        return None
    return [sections[i] for i in range(1, max_idx + 1)]


# ---------------------------------------------------------------------------
# Canned planner answers
# ---------------------------------------------------------------------------

GOOD_ANSWER = """=== GLOBAL ===
A pale woman in a dark ceremonial robe stands inside an ancient temple. Cold
metallic architecture, dim amber ritual light, realistic materials.

=== CLIPS ===
clip_1:
The woman walks toward the central altar. Her robe moves naturally with each
step. The crowd remains still and attentive.

clip_2:
She continues the same walk and gradually raises her right hand. The ritual
lights begin pulsing softly across the walls.

clip_3:
Her raised hand reaches the altar surface. The symbols activate and fill the
chamber with warm light.

=== CAMERAS ===
[
  {"clip_id": "clip-1", "clip_name": "Approach", "shot_size": "Medium Shot", "rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Natural 35mm", "stabilization": "Gimbal Smooth", "movement": "Track Forward", "speed": "Slow", "transition_type": "Continuous / Same Shot", "space_relation": "Same Space", "entity_continuity": "Lock Population / Layout", "transition_to_next": true},
  {"clip_id": "clip-2", "clip_name": "Reveal", "shot_size": "Medium Close-Up", "rig": "3-Axis Gimbal", "camera_body": "Cinematic Neutral", "lens": "Portrait 65mm", "stabilization": "Gimbal Smooth", "movement": "Push-In", "speed": "Slow", "transition_type": "Continuous / Same Shot", "space_relation": "Same Space", "entity_continuity": "Lock Population / Layout", "transition_to_next": true},
  {"clip_id": "clip-3", "clip_name": "Activation", "shot_size": "Close-Up", "rig": "Tripod / Locked Head", "camera_body": "Cinematic Neutral", "lens": "Portrait 85mm", "stabilization": "Hard Locked", "movement": "Locked-Off / Static", "speed": "Static", "transition_type": "Continuous / Same Shot", "space_relation": "Same Space", "entity_continuity": "Lock Population / Layout", "transition_to_next": false}
]
"""

DIALOGUE_ANSWER = """=== GLOBAL ===
Night train interior, warm reading light, realistic textures.

=== CLIPS ===
clip_1:
A young woman reads a folded letter, then looks up at the passing city lights
and quietly says <d>[Russian] Я всё решила. </d>

clip_2:
She continues watching the lights, calm and resolved.

=== CAMERAS ===
[{"clip_id": "clip-1", "clip_name": "", "shot_size": "Medium Close-Up", "rig": "Fluid Head Tripod", "camera_body": "Cinematic Neutral", "lens": "Natural 40mm", "stabilization": "Rig Native", "movement": "Locked-Off / Static", "speed": "Static", "transition_type": "Continuous / Same Shot", "space_relation": "Same Space", "entity_continuity": "Lock Population / Layout", "transition_to_next": true},
{"clip_id": "clip-2", "clip_name": "", "shot_size": "Medium Shot", "rig": "Fluid Head Tripod", "camera_body": "Cinematic Neutral", "lens": "Natural 40mm", "stabilization": "Rig Native", "movement": "Pan Right", "speed": "Ultra Slow", "transition_type": "Continuous / Same Shot", "space_relation": "Same Space", "entity_continuity": "Lock Population / Layout", "transition_to_next": false}]
"""

MESSY_ANSWER = """Sure! Here is your plan:

```text
=== GLOBAL ===
A quiet garden at dawn.

=== CLIPS ===
clip_1:
Mist drifts slowly across the pond.

clip_2:
Mist continues drifting as the sun rises.

=== CAMERAS ===
[{"clip_id":"clip-1","shot_size":"Wide Shot","movement":"Locked-Off / Static"},
{"clip_id":"clip-2","shot_size":"medium shot","movement":"Pan Left","transition_to_next":"true"}]
```
Hope this helps!"""


def main():
    # ---------------- light_clean ----------------
    cleaned = planner_mode.light_clean("```text\nA\n```")
    check("light_clean removes fence lines", "```" not in cleaned and "A" in cleaned)
    think_open = chr(60) + "think" + chr(62)
    think_close = chr(60) + "/think" + chr(62)
    cleaned = planner_mode.light_clean(
        "intro " + think_open + " Let me plan the sections carefully. " + think_close
        + " final answer")
    check("light_clean removes think blocks",
          "Let me plan" not in cleaned and "final answer" in cleaned)

    # ---------------- split_plan ----------------
    split = planner_mode.split_plan(GOOD_ANSWER)
    check("split: has all sections", split.has_global and split.has_clips and split.has_cameras)
    check("split: global text", "ceremonial robe" in split.global_text)
    check("split: clips text has 3 sections",
          "clip_1:" in split.clips_text and "clip_3:" in split.clips_text)
    check("split: cameras text is JSON array", split.cameras_text.strip().startswith("["))
    check("split: no warnings on the contract answer", not split.warnings,
          f"warnings={split.warnings}")

    split_m = planner_mode.split_plan(MESSY_ANSWER)
    check("split: fences stripped, sections found",
          split_m.has_global and split_m.has_clips and split_m.has_cameras)
    messy_cards, _mw = planner_mode.parse_cameras(split_m.cameras_text, 2)
    check("split: trailing chatter after cameras JSON is harmless",
          len(messy_cards) == 2 and "Hope this helps" not in json.dumps(messy_cards))

    no_global = planner_mode.split_plan("=== CLIPS ===\nclip_1: A\n\nclip_2: B\n\n=== CAMERAS ===\n[]")
    check("split: missing GLOBAL flagged", not no_global.has_global
          and any("GLOBAL" in w for w in no_global.warnings))

    out_of_order = planner_mode.split_plan("=== CLIPS ===\nclip_1: A\nclip_2: B\n\n=== GLOBAL ===\nG\n\n=== CAMERAS ===\n[]")
    check("split: out-of-order markers detected",
          any("canonical order" in w for w in out_of_order.warnings))
    check("split: out-of-order still slices by name",
          "G" in out_of_order.global_text and "clip_1" in out_of_order.clips_text)

    # heuristic fallback: no markers at all
    heur = planner_mode.split_plan(
        "A lone lighthouse in a storm.\n\nclip_1:\nThe beam sweeps left.\n\nclip_2:\nThe beam sweeps right.")
    check("split: heuristic global from prose", heur.has_global and "lighthouse" in heur.global_text)
    check("split: heuristic clips parsed", heur.has_clips and "clip_1" in heur.clips_text)
    check("split: heuristic warns", any("markers" in w for w in heur.warnings))

    # chatter preamble must NOT become global
    chatter = planner_mode.split_plan("Sure! Here it is:\nclip_1: A\nclip_2: B")
    check("split: chatter does not become GLOBAL", not chatter.has_global)

    # ---------------- duration arithmetic ----------------
    count, w = planner_mode.planned_clip_count(24.0, 8.0)
    check("count: 24s/8s -> 3", count == 3 and not w)
    count, w = planner_mode.planned_clip_count(30.0, 8.0)
    check("count: 30s/8s -> 4", count == 4)
    count, w = planner_mode.planned_clip_count(2400.0, 15.0)
    check("count: clamped to 16 with warning", count == 16 and any("clamped" in x for x in w))
    count, w = planner_mode.planned_clip_count(10.0, 8.0)
    check("count: minimum 2 clips", count == 2)

    durs, w = planner_mode.compute_clip_durations(24.0, 8.0)
    check("durations: 24s/3 -> [8,8,8]", durs == [8.0, 8.0, 8.0], f"got {durs}")
    durs, _ = planner_mode.compute_clip_durations(30.0, 8.0)
    check("durations: 30s/4 -> 7.5 each, sum=30",
          durs == [7.5] * 4 and abs(sum(durs) - 30.0) < 1e-9, f"got {durs}")
    durs, _ = planner_mode.compute_clip_durations(25.0, 8.0)
    check("durations: remainder on last clip, sum=25",
          abs(sum(durs) - 25.0) < 1e-9 and len(durs) == 4, f"got {durs}")
    durs, w = planner_mode.compute_clip_durations(2400.0, 15.0)
    check("durations: 16x150 for 2400s", durs == [150.0] * 16, f"got {durs}")
    durs, w = planner_mode.compute_clip_durations(24.0, 4.0, count=2)
    check("durations: explicit count honored", durs == [12.0, 12.0], f"got {durs}")

    # long clip targets (the clip_duration widget, up to 150s)
    count, w = planner_mode.planned_clip_count(60.0, 30.0)
    check("count: 60s/30s -> 2", count == 2 and not w, f"got {count}, {w}")
    durs, w = planner_mode.compute_clip_durations(60.0, 30.0)
    check("durations: 60s at 30s target -> [30,30]", durs == [30.0, 30.0], f"got {durs}")
    count, w = planner_mode.planned_clip_count(2400.0, 150.0)
    check("count: 2400s/150s -> 16, no clamp warning", count == 16 and not w)
    durs, w = planner_mode.compute_clip_durations(2400.0, 150.0)
    check("durations: 2400s at 150s target -> 16x150", durs == [150.0] * 16, f"got {durs}")
    count, w = planner_mode.planned_clip_count(100.0, 150.0)
    check("count: total below target -> minimum 2 clips", count == 2)
    durs, _ = planner_mode.compute_clip_durations(100.0, 150.0)
    check("durations: 100s at 150s target -> [50,50]", durs == [50.0, 50.0], f"got {durs}")

    # ---------------- parse_clips ----------------
    clips, w = planner_mode.parse_clips(split.clips_text, 3)
    check("clips: 3 parsed", len(clips) == 3, f"got {len(clips)}")
    check("clips: first body text", "altar" in clips[0])

    dup = "clip_1: A\nclip_2: B\nclip_2: B2\nclip_3: C"
    clips, w = planner_mode.parse_clips(dup, 3)
    check("clips: duplicate warned, first kept", "B" == clips[1]
          and any("duplicate clip_2" in x for x in w))

    gap = "clip_1: A\nclip_3: C"
    clips, w = planner_mode.parse_clips(gap, 3)
    check("clips: gap renumbered", len(clips) == 2 and any("gaps" in x for x in w))

    numbered = "1. Mist drifts.\n2. Sun rises."
    clips, w = planner_mode.parse_clips(numbered, 2)
    check("clips: numbered-list fallback", len(clips) == 2
          and any("numbered list" in x for x in w))

    yamlish = "clip_1:\nduration: 8\nprompt: |\n  She walks.\nseed: 5\n\nclip_2:\nduration: 8\nprompt: |\n  She stops."
    clips, w = planner_mode.parse_clips(yamlish, 2)
    check("clips: YAML wrappers stripped", clips and "duration" not in clips[0]
          and "seed" not in clips[0] and "She walks." in clips[0], f"got {clips}")

    inline = "clip_1: One line body.\nclip_2: Another body."
    clips, _ = planner_mode.parse_clips(inline, 2)
    check("clips: inline header text captured", "One line body." in clips[0])

    # ---------------- assemble + LongMedia import cross-check ----------------
    prompt = planner_mode.assemble_planner_prompt(clips)
    imported = lm_import_sections(prompt)
    check("assemble: LongMedia parser imports our format",
          imported == clips, f"imported={imported}")

    p2 = planner_mode.assemble_planner_prompt(
        planner_mode.parse_clips(split.clips_text, 3)[0])
    check("assemble: contract answer importable",
          lm_import_sections(p2) is not None)

    # dialogue survives verbatim through assemble + import
    dlg_split = planner_mode.split_plan(DIALOGUE_ANSWER)
    dlg_clips, dw = planner_mode.parse_clips(dlg_split.clips_text, 2)
    dlg_prompt = planner_mode.assemble_planner_prompt(dlg_clips)
    check("dialogue: <d> block survives verbatim",
          "<d>[Russian] Я всё решила. </d>" in dlg_prompt)
    check("dialogue: importable by LongMedia", lm_import_sections(dlg_prompt) is not None)

    # ---------------- cameras ----------------
    cards, w = planner_mode.parse_cameras(split.cameras_text, 3)
    check("cameras: 3 valid cards, no warnings", len(cards) == 3 and not w,
          f"warnings={w}")
    check("cameras: key order matches LongMedia card",
          all(tuple(c.keys()) == camera_knowledge.CAMERA_CARD_KEYS for c in cards))
    check("cameras: values exact", cards[0]["shot_size"] == "Medium Shot"
          and cards[1]["movement"] == "Push-In")
    check("cameras: last transition_to_next forced false",
          cards[-1]["transition_to_next"] is False and cards[0]["transition_to_next"] is True)
    check("cameras: clip ids sequential",
          [c["clip_id"] for c in cards] == ["clip-1", "clip-2", "clip-3"])

    messy_cards, mw = planner_mode.parse_cameras(split_m.cameras_text, 2)
    check("cameras: messy answer still yields 2 cards", len(messy_cards) == 2)
    check("cameras: missing fields defaulted with warnings",
          messy_cards[0]["rig"] == "Tripod / Locked Head"
          and any("missing 'rig'" in x for x in mw))
    check("cameras: 'medium shot' normalized to 'Medium Shot'",
          messy_cards[1]["shot_size"] == "Medium Shot"
          and any("normalized" in x for x in mw))
    check("cameras: string 'true' coerced to bool",
          isinstance(messy_cards[1]["transition_to_next"], bool))
    check("cameras: last card transition forced false even if model said true",
          messy_cards[1]["transition_to_next"] is False)

    bad = '[{"shot_size": "Impossible Framing", "movement": "Track Forward"}]'
    bad_cards, bw = planner_mode.parse_cameras(bad, 1)
    check("cameras: invalid value -> default + warning",
          bad_cards[0]["shot_size"] == "Medium Shot"
          and any("invalid value" in x for x in bw))

    pad_cards, pw = planner_mode.parse_cameras("[]", 3)
    check("cameras: empty array padded with defaults",
          len(pad_cards) == 3 and any("filled with defaults" in x for x in pw))

    trunc_cards, tw = planner_mode.parse_cameras(split.cameras_text, 2)
    check("cameras: extra cards truncated with warning",
          len(trunc_cards) == 2 and any("truncated" in x for x in tw))

    cams_json = planner_mode.cameras_to_json(cards)
    check("cameras_to_json: round-trips", json.loads(cams_json) == cards)
    check("cameras_to_json: empty -> '[]'", planner_mode.cameras_to_json([]) == "[]")

    # ---------------- camera_knowledge ----------------
    sizes = camera_knowledge.enum_sizes()
    check("knowledge: enum sizes mirror LongMedia",
          sizes == {"shot_size": 13, "rig": 25, "camera_body": 29, "lens": 32,
                    "stabilization": 10, "movement": 35, "speed": 8,
                    "transition_type": 4, "space_relation": 3, "entity_continuity": 3},
          f"got {sizes}")
    block = camera_knowledge.render_knowledge_block()
    check("knowledge: render contains every value",
          all(f'"{v}"' in block for allowed in camera_knowledge.ENUM_FIELDS.values()
              for v in allowed))
    resolved, note = camera_knowledge.resolve_enum_value("movement", "orbit clockwise")
    check("knowledge: fuzzy normalization", resolved == "Orbit Clockwise" and note != "exact")
    resolved, note = camera_knowledge.resolve_enum_value("lens", "Anamorphic 35mm")
    check("knowledge: exact match", resolved == "Anamorphic 35mm" and note == "exact")
    resolved, note = camera_knowledge.resolve_enum_value("speed", "")
    check("knowledge: empty -> None", resolved is None)

    # ---------------- prompt builders ----------------
    system, user = prompt_builder.build_planner_messages(
        24.0, 8.0, 3, [8.0, 8.0, 8.0], "a ritual in a temple", "", "keep it reverent", "", 0)
    check("planner system: knowledge injected",
          "Medium Shot" in system and "Orbit Clockwise" in system
          and "{{CAMERA_KNOWLEDGE_BLOCK}}" not in system)
    check("planner system: contract present",
          "=== GLOBAL ===" in system and "=== CLIPS ===" in system and "=== CAMERAS ===" in system)
    check("planner user: timing constraint",
          "24 seconds" in user and "3 consecutive clips" in user and "8s" in user)
    check("planner user: no pictures note", "No reference pictures" in user)
    check("planner user: idea + extras", "a ritual in a temple" in user and "reverent" in user)

    system2, user2 = prompt_builder.build_planner_messages(
        24.0, 8.0, 3, [8.0, 8.0, 8.0], "x", "A grey cat on a sofa.", "", "", 2)
    check("planner user: vision context + pictures note",
          "A grey cat on a sofa." in user2 and "2 reference picture(s)" in user2)

    system3, _ = prompt_builder.build_planner_messages(
        24.0, 8.0, 3, [8.0] * 3, "x", "", "", "MY CUSTOM SYSTEM", 0)
    check("planner system: custom override wins", system3 == "MY CUSTOM SYSTEM")

    tsys, tuser = prompt_builder.build_translate_messages("GLOBAL:\nG\n\nclip_1:\nA")
    check("translate: system file loaded", "GLOBAL:" in tsys and "verbatim" in tsys.lower())
    check("translate: user is the EN text", tuser == "GLOBAL:\nG\n\nclip_1:\nA")

    # ---------------- translation helpers ----------------
    src = planner_mode.translate_source("Global text.", ["Clip one.", "Clip two."])
    check("translate_source: shape",
          src == "GLOBAL:\nGlobal text.\n\nclip_1:\nClip one.\n\nclip_2:\nClip two.")
    ru, rw = planner_mode.parse_translated(
        "GLOBAL:\nГлобальный текст.\n\nclip_1:\nКлип один.\n\nclip_2:\nКлип два.")
    check("parse_translated: keeps labels", "clip_1:" in ru and "Клип один." in ru and not rw)
    ru_bad, rw_bad = planner_mode.parse_translated("Просто текст без меток.")
    check("parse_translated: lost labels -> empty + warning",
          ru_bad == "" and any("clip_N" in w for w in rw_bad))
    check("parse_translated: <d> preserved in mirror",
          "<d>[Russian]" in planner_mode.parse_translated(
              "GLOBAL:\nГ\n\nclip_1:\nОна говорит <d>[Russian] Да. </d> и уходит.")[0])

    # ---------------- reference_images_status ----------------
    check("ref status: none", json.loads(planner_mode.reference_images_status(None))
          == {"images_connected": False, "count": 0, "slots": []})
    check("ref status: dict slots",
          json.loads(planner_mode.reference_images_status(
              {"image_0": object(), "image_2": object(), "image_1": None}))
          == {"images_connected": True, "count": 2, "slots": ["image_0", "image_2"]})
    check("ref status: bare tensor", json.loads(planner_mode.reference_images_status(object()))
          == {"images_connected": True, "count": 1, "slots": ["image_0"]})

    # ---------------- token budgets ----------------
    check("tokens: planner min 2048", planner_mode.estimate_planner_tokens(1) == 2048)
    check("tokens: planner 3 clips still at the 2048 floor",
          planner_mode.estimate_planner_tokens(3) == 2048)
    check("tokens: planner 16 clips within cap",
          4096 <= planner_mode.estimate_planner_tokens(16) <= 8192)
    check("tokens: translate bounds",
          1024 <= planner_mode.estimate_translate_tokens("one two three") <= 8192)
    check("tokens: translate scales",
          planner_mode.estimate_translate_tokens(" ".join(["w"] * 4000)) == 8192)

    # ---------------- per-clip word budgets (long clips) ----------------
    b8 = planner_mode.per_clip_word_budget(8.0)
    b15 = planner_mode.per_clip_word_budget(15.0)
    b30 = planner_mode.per_clip_word_budget(30.0)
    b60 = planner_mode.per_clip_word_budget(60.0)
    b150 = planner_mode.per_clip_word_budget(150.0)
    check("budget: 8-15s clips stay in the default 50-90 band",
          50 <= b8 <= 90 and 50 <= b15 <= 90, f"got {b8}, {b15}")
    check("budget: monotonic with duration",
          b8 < b30 < b60 < b150, f"got {b8}, {b30}, {b60}, {b150}")
    check("budget: capped at 250", b150 == 250 and
          planner_mode.per_clip_word_budget(1000.0) == 250)
    check("budget: floor at 50", planner_mode.per_clip_word_budget(0.0) == 50
          and planner_mode.per_clip_word_budget(-5.0) == 50)
    check("tokens: estimate scales with long durations",
          planner_mode.estimate_planner_tokens(16, [150.0] * 16)
          > planner_mode.estimate_planner_tokens(16, [8.0] * 16))
    check("tokens: 16x150s plan still within the 8192 cap",
          2048 <= planner_mode.estimate_planner_tokens(16, [150.0] * 16) <= 8192)
    check("tokens: durations shorter than count padded with the floor",
          planner_mode.estimate_planner_tokens(4, [30.0])
          == planner_mode.estimate_planner_tokens(4, [30.0, 30.0, 30.0, 30.0]))

    # ---------------- import-safety of a long real-world-shaped plan ----------------
    long_clips = [f"Beat {i}: the subject advances through stage {i} while the "
                  f"environment evolves and the lighting warms." for i in range(1, 17)]
    long_prompt = planner_mode.assemble_planner_prompt(long_clips)
    imported = lm_import_sections(long_prompt)
    check("assemble: 16-clip plan importable", imported == long_clips)

    # ---------------- defuse of header-colliding body lines ----------------
    colliding = ("=== CLIPS ===\nclip_1:\nThe ritual begins.\nClip 2 continues the "
                 "action with a louder chant.\n\nclip_2:\nClip 1 echoes as the chant fades.")
    c_split = planner_mode.split_plan(colliding)
    c_clips, _cw = planner_mode.parse_clips(c_split.clips_text, 2)
    c_prompt = planner_mode.assemble_planner_prompt(c_clips)
    imported = lm_import_sections(c_prompt)
    check("defuse: colliding body lines neutralized and importable",
          imported is not None and len(imported) == 2
          and "louder chant" in imported[0] and "chant fades" in imported[1],
          f"imported={imported}")
    check("defuse: first-line collision gets 'And ' prefix",
          c_clips[1].startswith("And Clip 1 echoes"))

    print()
    if FAILURES:
        print(f"PLANNER TESTS FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PLANNER TESTS PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
