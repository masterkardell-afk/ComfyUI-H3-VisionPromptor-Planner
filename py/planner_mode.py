"""Planner mode pipeline for the LongMedia Planner integration.

Pure-python (no ComfyUI imports) so every function is unit-testable outside
ComfyUI. Responsibilities:

  - light_clean(): think-blocks / code-fence stripping WITHOUT the original
    H3 post_processor.finalize() (which would cut the text at H3 headers and
    auto-append audio fields — both fatal for a MultiClip plan).
  - split_plan(): deterministic split of the model answer into the three
    contract sections (=== GLOBAL === / === CLIPS === / === CAMERAS ===)
    with heuristic fallbacks and warnings.
  - planned_clip_count() / compute_clip_durations(): the arithmetic behind
    clip_durations (clip_count = ceil(total/target), clamp 2..16, <=150s per
    clip, even split with the rounding remainder on the last clip).
  - parse_clips(): LongMedia-importable clip_N: sections (the header grammar
    mirrors LongMedia's _v046 parser: clip_N:/shot_N:, markdown/bold labels,
    YAML-ish prompt: wrappers, numbered lists).
  - parse_cameras(): JSON camera cards validated against camera_knowledge.
  - reference_images_status(): machine-readable marker of connected images.
  - scrub_photo_meta() (v1.1.7): photo-description leak scrubber — when a
    small instruct-VLM copies the reference-image analysis into the plan
    ("The photo shows ..." / "На фото изображено ..."), tier-1 sentences
    (explicit source-image references) are removed from every section and
    tier-2 sentences (generic photo wording) from the GLOBAL section only,
    each with a loud warning; clean text passes through byte-identical.
  - estimate_planner_tokens() / estimate_translate_tokens(): token budgets
    (planner mode requests its own budget, independent of the max_tokens
    widget).
  - Creative boost (fork v1.1.5/v1.1.6): normalize_creative_boost(), the built-in visual
    motif pools (world spine + per-clip beats, deterministic per seed), the
    CREATIVE WRITING STANDARDS system-prompt block, and expressive-level word
    budgets — everything the fork injects to keep plans vivid and varied
    WITHOUT breaking the LongMedia multiclip continuity rules (level 'off'
    renders to empty strings: no creative scaffolding is injected at all).
"""

from __future__ import annotations

import json
import math
import random
import re

from . import camera_knowledge

MIN_CLIPS = 2
MAX_CLIPS = 16
MAX_CLIP_SECONDS = 150.0
MIN_TOTAL_SECONDS = 8.0
MAX_TOTAL_SECONDS = 2400.0

# Per-clip narrative word budget: the generation contract's default is
# "50-90 words" for a standard ~8-15s clip. Longer clips (the planner target
# may be up to MAX_CLIP_SECONDS via the clip_duration widget) get a budget
# that grows sub-linearly with the clip's seconds, clamped to a paragraph-ish
# maximum (250 words keeps 16x150s plans inside the 8192-token cap).
WORD_BUDGET_BASE = 45.0
WORD_BUDGET_PER_SECOND = 1.4
WORD_BUDGET_MIN = 50
WORD_BUDGET_MAX = 250
# Clips at or below this length keep the contract's default "50-90 words"
# rule; the user message only spells out explicit budgets above it.
STANDARD_CLIP_SECONDS = 15.5

# ---------------------------------------------------------------------------
# Creative boost (fork v1.1.5)
# ---------------------------------------------------------------------------
# Why: the planner contract is 49 lines of rules and prohibitions and not a
# single line asking for vivid, varied writing — instruct models answer that
# with safe template prose ("the subject continues moving..."), identical
# sentence patterns and the same lighting in every clip. The levels below
# inject creative standards + per-clip motif ingredients on top of the
# untouched contract:
#   off        -> nothing injected, byte-identical planner prompts (v1.1.4)
#   standard   -> CREATIVE WRITING STANDARDS block in the system prompt
#   expressive -> standard + per-clip sampled motif hints in the user message
#                 + a wider (still contract-compatible) word budget band
CREATIVE_LEVELS = ("off", "standard", "expressive")
CREATIVE_LEVEL_DEFAULT = "standard"

# Expressive word budget: a denser band that still lands INSIDE the
# contract's default "50-90 words" for standard 8-15s clips (72..84 words)
# and keeps the 250-word cap for long clips (16x150s plans must stay within
# the 8192-token generation budget).
EXPRESSIVE_WORD_BUDGET_BASE = 58.0
EXPRESSIVE_WORD_BUDGET_PER_SECOND = 1.75

# Section markers of the generation contract.
SECTION_NAMES = ("GLOBAL", "CLIPS", "CAMERAS")
SECTION_MARKER_RE = re.compile(
    r"^[ \t]*(?:={2,}|#{1,6}|\*{2,}|-{3,})[ \t]*(GLOBAL|CLIPS|CAMERAS)[ \t]*"
    r"(?:={2,}|#{1,6}|\*{2,}|-{3,})[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.MULTILINE)

# LongMedia _V046_MULTICLIP_HEADER_RE (nodes.py) — mirrored so our clip
# sections are guaranteed to be importable by the LongMedia Planner.
_CLIP_HEADER_RE = re.compile(
    r"^\[?(?:clip|shot)[ _-]*(\d{1,2})\]?(?:\s*\([^)]*\))?\s*(?:(?::|=|[-–—])\s*)?(.*)$",
    re.IGNORECASE,
)
_NUMBERED_RE = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(.+?)\s*$")
_YAML_PROMPT_RE = re.compile(r"^\s*(?:[-*+]\s*)?prompt\s*:\s*(?:[|>][-+]?)?\s*(.*)$", re.IGNORECASE)
_YAML_META_RE = re.compile(r"^\s*(?:duration|seed)\s*:\s*.*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def light_clean(raw) -> str:
    """Strip <think> blocks and standalone code-fence lines; normalize EOLs.

    Deliberately does NOT cut preambles or append anything — the section
    split below is the only structural operation applied.
    """
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _THINK_RE.sub("", text)
    text = _FENCE_LINE_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section split
# ---------------------------------------------------------------------------

class PlanSplit:
    """Result of splitting a model answer into contract sections."""

    __slots__ = ("global_text", "clips_text", "cameras_text", "warnings",
                 "has_global", "has_clips", "has_cameras")

    def __init__(self, global_text="", clips_text="", cameras_text="",
                 warnings=None, has_global=False, has_clips=False, has_cameras=False):
        self.global_text = global_text
        self.clips_text = clips_text
        self.cameras_text = cameras_text
        self.warnings = list(warnings or [])
        self.has_global = has_global
        self.has_clips = has_clips
        self.has_cameras = has_cameras

    @property
    def ok(self) -> bool:
        return self.has_clips


def split_plan(raw) -> PlanSplit:
    """Split a raw model answer into GLOBAL / CLIPS / CAMERAS sections.

    Primary path: the === SECTION === markers of the generation contract.
    Fallback path (markers missing): sniff a JSON camera array, then clip_N:
    headers, then treat leading prose as the global section — every fallback
    step emits a warning.
    """
    warnings: list[str] = []
    text = light_clean(raw)
    if not text:
        return PlanSplit(warnings=["planner answer is empty"])

    markers = [(m.group(1).upper(), m.start(), m.end())
               for m in SECTION_MARKER_RE.finditer(text)]

    if markers:
        seen = {name for name, _s, _e in markers}
        for name in SECTION_NAMES:
            if name not in seen:
                warnings.append(f"section === {name} === missing from the model answer")
        # Duplicate markers: keep the first occurrence, warn about the rest.
        by_name: dict[str, tuple[int, int]] = {}
        for name, s, e in markers:
            if name in by_name:
                warnings.append(f"duplicate === {name} === marker; kept the first one")
            else:
                by_name[name] = (s, e)

        global_text, clips_text, cameras_text = "", "", ""
        has_global = has_clips = has_cameras = False
        items = sorted(by_name.items(), key=lambda kv: kv[1])
        for i, (name, (_s, e)) in enumerate(items):
            seg_end = items[i + 1][1][0] if i + 1 < len(items) else len(text)
            body = text[e:seg_end].strip("\n").strip()
            if name == "GLOBAL":
                global_text, has_global = body, bool(body)
            elif name == "CLIPS":
                clips_text, has_clips = body, bool(body)
            elif name == "CAMERAS":
                cameras_text, has_cameras = body, bool(body)
        if [n for n, _ in items] != [n for n in SECTION_NAMES if n in by_name]:
            warnings.append("section markers appeared out of the canonical order "
                            "(GLOBAL -> CLIPS -> CAMERAS)")
        return PlanSplit(global_text, clips_text, cameras_text,
                         warnings, has_global, has_clips, has_cameras)

    # ---- heuristic fallback (no contract markers) ----
    warnings.append("no === SECTION === markers found; using heuristic split")
    cameras_text, cameras_end = _sniff_json_array(text)
    if cameras_text:
        warnings.append("cameras JSON sniffed from the answer body")
        text = text[:cameras_end].strip("\n").strip()

    sections = _clip_header_sections(text)
    if sections:
        first_pos = _first_header_position(text)
        global_candidate = text[:first_pos].strip("\n").strip() if first_pos else ""
        # Prose before the first clip header only counts as GLOBAL when it is
        # short and does not look like model chatter (braces/fences/prefixes).
        if global_candidate and len(global_candidate) < 1200 and not re.match(
                r"^(sure|here|certainly|of course|okay|ok)[,.! ]", global_candidate, re.I):
            return PlanSplit(global_candidate, text[first_pos:].strip("\n").strip(),
                             cameras_text, warnings, True, True, bool(cameras_text))
        return PlanSplit("", text.strip(), cameras_text, warnings, False, True, bool(cameras_text))

    return PlanSplit("", "", cameras_text, warnings, False, False, bool(cameras_text))


def _sniff_json_array(text: str) -> tuple[str, int]:
    """Best-effort extraction of a JSON array of camera cards from free text."""
    first = text.find("[")
    last = text.rfind("]")
    if first == -1 or last <= first:
        return "", 0
    candidate = text[first:last + 1]
    try:
        value = json.loads(candidate)
    except Exception:
        return "", 0
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        keys = set(value[0].keys())
        if keys & set(camera_knowledge.CAMERA_CARD_KEYS):
            return candidate, first
    if isinstance(value, list):
        return "", 0
    return "", 0


def _clip_header_line(line: str):
    """Mirror of LongMedia's _v046_header preprocessing + regex."""
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", str(line or "")).strip()
    s = re.sub(r"^\s*(?:[-+]\s+)(?=(?:\*\*)?\[?(?:clip|shot)\b)", "", s, flags=re.I)
    s = s.replace("**", "").replace("`", "").strip()
    m = _CLIP_HEADER_RE.match(s)
    if not m:
        return None
    idx = int(m.group(1))
    if idx < 1 or idx > MAX_CLIPS:
        return None
    return idx, str(m.group(2) or "").strip()


def _clip_header_sections(text: str) -> dict[int, str]:
    """clip_N:-style sections -> {index: body}.

    Two-pass strategy (the fork's contract tells the model to emit bare
    'clip_N:' labels on their own lines):

      Pass 1 (preferred): when at least two BARE labels (empty inline text,
      e.g. 'clip_2:') exist, ONLY those lines split sections. Body lines that
      accidentally collide with the header grammar (e.g. a wrapped line
      starting with 'Clip 3 ...') stay inside their section, where
      _defuse_body neutralizes them afterwards.

      Pass 2 (fallback): the greedy LongMedia grammar (inline text after the
      label, markdown variants) — for answers written as
      'clip_1: text' single-line style.
    """
    lines = text.split("\n")

    def bare(line):
        h = _clip_header_line(line)
        if h is not None and not h[1].strip():
            return h[0]
        return None

    bare_at: dict[int, int] = {}
    for i, line in enumerate(lines):
        idx = bare(line)
        if idx is not None and i not in bare_at:
            bare_at[i] = idx

    if len(set(bare_at.values())) >= 2:
        sections: dict[int, str] = {}
        current = None
        body: list[str] = []

        def flush():
            nonlocal current, body
            if current is None or current in sections:
                body = []
                return
            sections[current] = _prompt_body(body)
            body = []

        for i, line in enumerate(lines):
            if i in bare_at:
                flush()
                current = bare_at[i]
            elif current is not None:
                body.append(line)
        flush()
        return sections

    # ---- greedy fallback (LongMedia-compatible) ----
    sections = {}
    current = None
    body = []

    def flush():
        nonlocal current, body
        if current is None or current in sections:
            body = []
            return
        sections[current] = _prompt_body(body)
        body = []

    for line in lines:
        header = _clip_header_line(line)
        if header is not None:
            flush()
            current, inline = header
            body = [inline] if inline else []
        elif current is not None:
            body.append(line)
    flush()
    return sections


def _defuse_body(body: str) -> str:
    """Neutralize body lines that would re-trigger the LongMedia clip/shot
    header grammar (e.g. a wrapped paragraph line starting with 'Clip 3 ...').

    The LongMedia import parser treats ANY line starting with clip/shot + a
    number as a NEW section header, silently re-splitting (or corrupting) the
    imported texts. Continuation lines that collide with the grammar are
    merged into their predecessor; a leading colliding line gets an 'And '
    prefix. Applied inside parse_clips so every downstream consumer
    (planner_prompt assembly, translation source) receives safe bodies.
    """
    out: list[str] = []
    for line in str(body or "").split("\n"):
        if _clip_header_line(line) is not None:
            if out and out[-1].strip():
                out[-1] = out[-1].rstrip() + " " + line.strip()
                continue
            out.append("And " + line.strip())
        else:
            out.append(line)
    return "\n".join(out).strip()


def _prompt_body(lines) -> str:
    """Light mirror of LongMedia's _v046_prompt_body: strip blank edges and
    YAML-ish prompt:/duration:/seed: wrappers."""
    body = [str(x) for x in (lines or [])]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return ""
    cleaned: list[str] = []
    for line in body:
        if _YAML_META_RE.match(line):
            continue
        m = _YAML_PROMPT_RE.match(line)
        if m:
            cleaned.append(m.group(1))
            continue
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _first_header_position(text: str) -> int:
    for i, line in enumerate(text.split("\n")):
        if _clip_header_line(line) is not None:
            return sum(len(l) + 1 for l in text.split("\n")[:i])
    return 0


# ---------------------------------------------------------------------------
# Duration arithmetic
# ---------------------------------------------------------------------------

def planned_clip_count(total_duration: float, target_clip_duration: float) -> tuple[int, list[str]]:
    """clip_count = ceil(total / target), clamped to the LongMedia range 2..16."""
    warnings: list[str] = []
    total = float(total_duration)
    target = float(target_clip_duration)
    if total < MIN_TOTAL_SECONDS or total > MAX_TOTAL_SECONDS:
        warnings.append(f"total_duration {total} outside {MIN_TOTAL_SECONDS}..{MAX_TOTAL_SECONDS}s; "
                        f"clamped for planning")
        total = min(max(total, MIN_TOTAL_SECONDS), MAX_TOTAL_SECONDS)
    if target <= 0:
        warnings.append("target clip duration must be > 0; using 8.0s")
        target = 8.0
    count = int(math.ceil(total / target)) if target > 0 else MIN_CLIPS
    if count < MIN_CLIPS:
        count = MIN_CLIPS
    elif count > MAX_CLIPS:
        count = MAX_CLIPS
        warnings.append(f"clip_count clamped to {MAX_CLIPS} (LongMedia maximum); "
                        f"per-clip duration grows accordingly")
    return count, warnings


def compute_clip_durations(total_duration: float, target_clip_duration: float,
                           count: int | None = None) -> tuple[list[float], list[str]]:
    """Even split of the total across clips; rounding remainder on the last clip.

    Each duration is <= 150s (LongMedia limit) and > 0.
    """
    warnings: list[str] = []
    if count is None:
        count, w = planned_clip_count(total_duration, target_clip_duration)
        warnings.extend(w)
    count = max(1, int(count))
    total = float(total_duration)
    base = round(total / count, 2)
    if base <= 0:
        base = round(MIN_TOTAL_SECONDS / count, 2)
    durations = [base] * count
    durations[-1] = round(total - base * (count - 1), 2)
    if durations[-1] <= 0:
        durations = [round(total / count, 2)] * count
        durations[-1] = round(total - durations[0] * (count - 1), 2)
    for i, d in enumerate(durations):
        if d > MAX_CLIP_SECONDS:
            durations[i] = MAX_CLIP_SECONDS
            warnings.append(f"clip {i + 1} duration capped at {MAX_CLIP_SECONDS}s")
        if d < 2.0:
            warnings.append(f"clip {i + 1} duration {d}s is very short")
    return durations, warnings


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def parse_clips(clips_text: str, expected_count: int) -> tuple[list[str], list[str]]:
    """Parse clip_N: sections into a clean, contiguous list of clip bodies.

    Guarantees: no duplicates, contiguous 1..N numbering, YAML wrappers
    stripped, numbered-list fallback applied. The count may differ from
    expected_count (the caller reconciles durations/cameras to the actual
    count and records a warning).
    """
    warnings: list[str] = []
    text = light_clean(clips_text)
    if not text:
        return [], ["clips section is empty"]

    sections = _clip_header_sections(text)
    if not sections:
        # numbered-list fallback: "1. ...", "2. ..."
        numbered: dict[int, str] = {}
        for line in text.split("\n"):
            m = _NUMBERED_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            if 1 <= idx <= MAX_CLIPS and idx not in numbered:
                numbered[idx] = str(m.group(2)).strip()
        if numbered:
            sections = numbered
            warnings.append("clips parsed from a numbered list (no clip_N: labels)")
        if not sections:
            return [], ["no clip_N: sections found in the clips section"]

    # _clip_header_sections keeps the FIRST body of every index; count raw
    # header occurrences to report duplicates honestly.
    seen: dict[int, int] = {}
    for line in text.split("\n"):
        header = _clip_header_line(line)
        if header is not None:
            seen[header[0]] = seen.get(header[0], 0) + 1
    for idx, n in sorted(seen.items()):
        if n > 1:
            warnings.append(f"duplicate clip_{idx} section ({n} occurrences); kept the first one")

    ordered = sorted(sections.keys())
    if ordered != list(range(1, len(ordered) + 1)):
        warnings.append("clip numbering had gaps; renumbered sequentially")

    clips = [_defuse_body(sections[idx]).strip() for idx in ordered if sections[idx].strip()]
    if len(clips) != expected_count:
        warnings.append(f"model produced {len(clips)} clip(s); plan expected {expected_count}")
    if len(clips) < MIN_CLIPS:
        warnings.append(f"only {len(clips)} clip(s); LongMedia Planner requires at least 2")
    return clips, warnings


def assemble_planner_prompt(clips) -> str:
    """Assemble the LongMedia-importable multiclip prompt: 'clip_N:\\n<body>'."""
    parts = []
    for i, body in enumerate(clips, start=1):
        parts.append(f"clip_{i}:\n{str(body).strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def parse_cameras(cameras_text: str, count: int, clip_names=None) -> tuple[list[dict], list[str]]:
    """Parse + validate the CAMERAS JSON into LongMedia camera cards.

    Returns (cards, warnings). cards has exactly `count` entries with
    CAMERA_CARD_KEYS ordering, sequential clip_ids and transition_to_next
    forced to False on the last card.
    """
    warnings: list[str] = []
    text = light_clean(cameras_text)
    if not text:
        return [], ["cameras section is empty; camera_params output is '[]'"]

    first = text.find("[")
    last = text.rfind("]")
    if first == -1 or last <= first:
        return [], ["cameras section does not contain a JSON array; camera_params output is '[]'"]
    candidate = text[first:last + 1]
    try:
        value = json.loads(candidate)
    except Exception as e:
        return [], [f"cameras JSON failed to parse ({e}); camera_params output is '[]'"]

    if isinstance(value, dict) and isinstance(value.get("cameras"), list):
        warnings.append("cameras JSON was an object with a 'cameras' key; using its list")
        value = value.get("cameras")
    if not isinstance(value, list):
        return [], ["cameras JSON is not an array; camera_params output is '[]'"]

    names = list(clip_names or [])
    cards: list[dict] = []
    for i, item in enumerate(value):
        card, w = camera_knowledge.normalize_card(item, i)
        if i < len(names) and names[i]:
            card["clip_name"] = str(names[i])[:120]
        warnings.extend(w)
        cards.append(card)

    if len(cards) < count:
        while len(cards) < count:
            card, w = camera_knowledge.normalize_card({}, len(cards))
            warnings.append(f"camera card {len(cards) + 1} missing; filled with defaults")
            warnings.extend(w)
            cards.append(card)
    elif len(cards) > count:
        warnings.append(f"cameras JSON had {len(cards)} cards; truncated to {count}")
        cards = cards[:count]

    if cards:
        cards[-1]["transition_to_next"] = False
    return cards, warnings


def cameras_to_json(cards) -> str:
    if not cards:
        return "[]"
    return json.dumps(cards, ensure_ascii=False, separators=(", ", ": "))


# ---------------------------------------------------------------------------
# Translation pass helpers
# ---------------------------------------------------------------------------

def translate_source(global_text: str, clips) -> str:
    """Assemble the EN text sent to the translation pass (GLOBAL: + clip_N:)."""
    parts = []
    g = str(global_text or "").strip()
    if g:
        parts.append("GLOBAL:\n" + g)
    for i, body in enumerate(clips, start=1):
        parts.append(f"clip_{i}:\n{str(body).strip()}")
    return "\n\n".join(parts)


def parse_translated(raw) -> tuple[str, list[str]]:
    """Light-clean the translated mirror and verify it kept the clip labels.

    Returns (text, warnings). text is "" when the structure was lost.
    """
    warnings: list[str] = []
    text = light_clean(raw)
    if not text:
        return "", ["translation pass returned an empty answer"]
    if not _clip_header_sections(text):
        # The mirror must keep clip_N: labels verbatim; without them the RU
        # text cannot be used in the same shape as the EN one.
        return "", ["translation lost the clip_N: labels; planner_prompt_ru left empty"]
    return text, warnings


# ---------------------------------------------------------------------------
# Reference images status
# ---------------------------------------------------------------------------

def reference_images_status(images) -> str:
    """Compact JSON marker of the connected reference images (no paths).

    ComfyUI passes IMAGE tensors, not file paths, so the marker reports the
    presence, count and slot names only.
    """
    slots: list[str] = []
    if images is None:
        pass
    elif isinstance(images, dict):
        slots = sorted(k for k in images.keys() if images.get(k) is not None)
    elif isinstance(images, (list, tuple)):
        slots = [f"image_{i}" for i, v in enumerate(images) if v is not None]
    else:
        slots = ["image_0"]
    return json.dumps({
        "images_connected": bool(slots),
        "count": len(slots),
        "slots": slots,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Photo-meta leak scrubber (v1.1.7)
# ---------------------------------------------------------------------------

# Tier 1 — explicit references to the SOURCE/attached image (EN + RU):
# stripped in EVERY section (GLOBAL, clips, RU mirror).
_PHOTO_META_SOURCE_RES = [re.compile(p, re.I) for p in (
    r"\b(?:the|this|these|those)\s+(?:reference|attached|provided|given|source|input)\s+"
    r"(?:photo(?:graph)?s?|pictures?|images?|references?)\b",
    r"\bas\s+(?:shown|depicted|seen)\s+in\s+(?:the\s+)?(?:reference\s+)?"
    r"(?:photo(?:graph)?|picture|image)\b",
    r"\bin\s+(?:the\s+)?(?:reference|attached|provided|source)\s+"
    r"(?:photo(?:graph)?|picture|image)\b",
    r"\b(?:the|this)\s+image\s+analysis\b",
    r"\breference\s+image\s+analysis\b",
    # RU
    r"\bна\s+(?:этом\s+|данном\s+)?(?:фото|фотографии|картинке|изображении|снимке)\b",
    r"\b(?:изображён\w*|изображен\w*|видн\w*)\s+на\s+"
    r"(?:фото|фотографии|картинке|изображении|снимке)\b",
    r"\b(?:референсн\w*|прилагаем\w*|предоставленн\w*)\s+"
    r"(?:изображени\w*|фото|фотографи\w*|картинк\w*|референс\w*)\b",
)]

# Tier 2 — generic photo-description wording: stripped from the GLOBAL
# section only; inside clip bodies a photograph can be a legitimate
# diegetic prop ("she picks up an old photograph"), so there it only
# produces a suspect warning.
_PHOTO_META_GENERIC_RES = [re.compile(p, re.I) for p in (
    r"\b(?:the|this|a|an)\s+(?:photo(?:graph)?|picture|image)\s+"
    r"(?:shows|depicts|captures|features|portrays|displays|presents)\b",
    r"\b(?:in|on)\s+(?:the\s+)?(?:photo(?:graph)?|picture|image)\b",
    r"\blooking\s+(?:at|into)\s+(?:the\s+)?camera\b",
    r"\b(?:poses|posing|posed)\s+(?:for|in)\b",
    r"\b(?:left|right)\s+(?:side|half|portion)\s+of\s+the\s+"
    r"(?:frame|photo(?:graph)?|image|picture)\b",
)]

_LABEL_LINE_RE = re.compile(r"^\s*(?:GLOBAL:|clip_\d+:)\s*$", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?:…])\s+")


def _photo_meta_hit(text: str) -> bool:
    return (any(p.search(text) for p in _PHOTO_META_SOURCE_RES)
            or any(p.search(text) for p in _PHOTO_META_GENERIC_RES))


def scrub_photo_meta(text: str, strict: bool = False,
                     label: str = "clip") -> tuple[str, list[str], int]:
    """Remove photo-description leakage from a plan section (v1.1.7).

    The generation contract forbids describing the source photograph, but a
    small instruct-VLM can still copy the reference-image analysis into the
    plan (the user-reported "На фото изображено ... и полный вижн рефа"
    GLOBAL). This is the deterministic backstop:

      - Tier 1 (explicit source-image references, EN + RU) — whole sentences
        removed in EVERY section (GLOBAL, clips, RU mirror).
      - Tier 2 (generic photo wording like "the photo shows" / "in the
        image") — whole sentences removed only when ``strict=True`` (the
        GLOBAL section); inside clip bodies it stays with a suspect warning
        because a photograph can be a diegetic prop there.

    Label lines (``GLOBAL:`` / ``clip_N:``) and ``<Picture k>`` labels are
    never touched. Clean input returns byte-identical text with no warnings
    and no count, so the scrub is a no-op for compliant plans (and for every
    canned answer without leakage — regression-safe). Returns
    (scrubbed_text, warnings, removed_sentence_count).
    """
    src = str(text or "")
    if not src.strip() or not _photo_meta_hit(src):
        return src, [], 0

    warnings: list[str] = []
    removed: list[str] = []
    suspects: list[str] = []
    out_lines: list[str] = []

    for line in src.split("\n"):
        if not line.strip() or _LABEL_LINE_RE.match(line):
            out_lines.append(line)
            continue
        line_t1 = any(p.search(line) for p in _PHOTO_META_SOURCE_RES)
        line_t2 = any(p.search(line) for p in _PHOTO_META_GENERIC_RES)
        if not line_t1 and not (line_t2 and strict):
            if line_t2:
                suspects.append(line.strip()[:120])
            out_lines.append(line)
            continue
        kept: list[str] = []
        line_removed = 0
        for sent in _SENT_SPLIT_RE.split(line):
            s_t1 = any(p.search(sent) for p in _PHOTO_META_SOURCE_RES)
            s_t2 = any(p.search(sent) for p in _PHOTO_META_GENERIC_RES)
            if s_t1 or (s_t2 and strict):
                removed.append(sent.strip())
                line_removed += 1
            else:
                if s_t2:
                    suspects.append(sent.strip()[:120])
                kept.append(sent)
        out_lines.append(" ".join(kept) if line_removed else line)

    out = "\n".join(out_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip("\n")

    if removed:
        example = removed[0][:100]
        warnings.append(
            f"photo-meta leak scrubbed from {label} (v1.1.7): {len(removed)} sentence(s) "
            f"removed, e.g. {example!r} — the plan must describe the video, never the "
            "source photograph; re-seed if too much was lost")
    if suspects:
        warnings.append(
            f"photo-meta suspect left in {label} (v1.1.7): photo-description wording "
            f"({len(suspects)} hit(s), e.g. {suspects[0][:100]!r}) kept — it may be a "
            "diegetic photograph prop; re-seed or edit the clip manually if unintended")
    return out, warnings, len(removed)


# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------

def _round_up64(value: int) -> int:
    return int(math.ceil(max(0, value) / 64.0)) * 64


def per_clip_word_budget(clip_seconds: float, creative_boost: str = "off") -> int:
    """Narrative word budget for ONE clip body, scaled with its duration.

    45 + 1.4*seconds, clamped to 50..250 words. An 8-15s clip lands inside
    the contract's default "50-90 words" band; a 60s clip targets ~129 words;
    a 150s clip is capped at 250 (16 such clips still fit the 8192-token
    generation budget).

    creative_boost='expressive' (v1.1.5) swaps in the denser band
    58 + 1.75*seconds (standard clips -> 72..84 words, still inside the
    contract band; the 250-word cap is unchanged).
    """
    seconds = max(0.0, float(clip_seconds or 0.0))
    if normalize_creative_boost(creative_boost)[0] == "expressive":
        base, per_second = EXPRESSIVE_WORD_BUDGET_BASE, EXPRESSIVE_WORD_BUDGET_PER_SECOND
    else:
        base, per_second = WORD_BUDGET_BASE, WORD_BUDGET_PER_SECOND
    words = base + per_second * seconds
    return int(min(WORD_BUDGET_MAX, max(WORD_BUDGET_MIN, round(words))))


def estimate_planner_tokens(clip_count: int, durations=None, creative_boost: str = "off") -> int:
    """Token budget for the single planner generation pass.

    ~1.5 tokens per word of clip body + ~130 tokens per camera card + global
    + section overhead + safety margin. With `durations` given, per-clip word
    budgets scale with each clip's seconds (long clips -> longer bodies);
    without it, the historical per-clip flat estimate (~260 tokens) applies.
    creative_boost='expressive' uses the denser word band (v1.1.5).
    """
    count = max(1, int(clip_count))
    if durations:
        per_clip = [per_clip_word_budget(d, creative_boost) for d in durations[:count]
                    ] + [WORD_BUDGET_MIN] * max(0, count - len(durations))
        estimate = 360 + sum(int(w * 1.5) + 130 for w in per_clip)
    else:
        estimate = 360 + count * 260
    return int(min(8192, max(2048, _round_up64(estimate))))


def estimate_translate_tokens(text: str) -> int:
    """Token budget for the EN->RU mirror pass (RU words tokenize heavier)."""
    words = len(str(text or "").split())
    estimate = int(words * 2.2) + 300
    return int(min(8192, max(1024, _round_up64(estimate))))


# ---------------------------------------------------------------------------
# Creative boost (fork v1.1.5)
# ---------------------------------------------------------------------------

def normalize_creative_boost(value) -> tuple:
    """Validate a creative_boost widget value.

    Returns (level, warning_or_None). Unknown / empty values fall back to
    CREATIVE_LEVEL_DEFAULT ('standard') with a warning instead of failing the
    run — creativity scaffolding must never break the planner pipeline.
    """
    text = str(value or "").strip().lower()
    if text in CREATIVE_LEVELS:
        return text, None
    if not text:
        return CREATIVE_LEVEL_DEFAULT, None
    return (CREATIVE_LEVEL_DEFAULT,
            f"creative_boost='{value}' is not one of {list(CREATIVE_LEVELS)}; "
            f"falling back to '{CREATIVE_LEVEL_DEFAULT}'")


# Built-in visual motif pools for expressive mode. STRICTLY in-world imagery:
# no camera/framing/lens/movement vocabulary — clip bodies must stay free of
# it, so the ingredients we hand the model must obey the same law.
MOTIF_POOLS = {
    "light": (
        "hard noon light carving sharp shadows",
        "a single practical lamp in near-darkness",
        "backlit dust drifting in a shaft of light",
        "neon bleeding through wet glass",
        "overcast pearl-grey light",
        "firelight flickering from below",
        "cold monitor glow on skin",
        "moonlight silvering every edge",
        "light dying mid-scene",
        "sun strobing through passing structures",
    ),
    "atmosphere": (
        "steam coiling low over the ground",
        "thin rain beading on every surface",
        "dry heat shimmering the distance",
        "snow falling in slow spirals",
        "pollen suspended in gold light",
        "smoke sinking like liquid",
        "sea spray hanging in the air",
        "dust raised by an unseen wind",
        "condensation creeping across glass",
        "petals falling like confetti",
    ),
    "texture": (
        "cracked paint peeling in curls",
        "oxidized copper gone green",
        "wet asphalt mirroring the sky",
        "raw silk catching the air",
        "frosted glass softening every shape",
        "grease-stained steel",
        "velvet swallowing the light",
        "moss reclaiming stone",
        "chrome polished to liquid",
        "worn leather cracked like dry earth",
    ),
    "color": (
        "muted teal and amber palette",
        "a monochrome world with one red object",
        "bleached highlights over ink-black shadows",
        "tungsten warmth against blue dusk",
        "pastel haze like a faded photograph",
        "saturated primaries dimmed by fog",
        "sepia dust over everything",
        "cold fluorescent light turning skin grey",
    ),
    "world_motion": (
        "a sudden stillness after motion",
        "fabric blooming in slow air",
        "water snapping back to mirror calm",
        "a crowd surging then parting",
        "everything trembling at one frequency",
        "one slow gesture inside urgent motion",
        "objects sliding into new arrangements",
        "ripples undoing a reflection",
    ),
    "story_beat": (
        "a discovery moment",
        "an approach that almost stops",
        "a release of held breath",
        "a reversal of who leads",
        "something left behind on purpose",
        "a repetition that differs by one detail",
        "an ending that echoes the beginning, changed",
        "a threshold crossed",
    ),
}

# Deterministic motif sampling (v1.1.6) — a two-level scheme that follows
# the LongMedia continuity law instead of fighting it:
#   - WORLD SPINE: two ingredients sampled ONCE for the whole plan (from the
#     light/atmosphere/texture/color pools) — the shared world state that
#     every clip must keep present and let evolve (repo rule: the Global
#     Prompt describes what remains constant; per-clip ingredients must not
#     re-dress the world from scratch every clip).
#   - PER-CLIP BEAT: one ingredient per clip (world_motion/story_beat pools)
#     — the change this clip contributes to the ongoing scene, with a short
#     no-repeat memory so neighbouring beats differ.
# Same seed -> same hints (reproducible plans); different seeds -> different
# creative directions.
WORLD_POOLS = ("light", "atmosphere", "texture", "color")
BEAT_POOLS = ("world_motion", "story_beat")


def sample_motif_hints(clip_count: int, seed: int) -> dict:
    """Sample the world spine (2 shared ingredients) + one beat per clip.

    Returns {"world": (a, b), "clips": {1: (beat,), ...}} — deterministic in
    `seed`. The spine is shared by every clip; each clip's single beat is
    drawn from the change pools (world_motion / story_beat) so it reads as
    development of the same scene, not a new set dressing.
    """
    count = max(0, int(clip_count))
    rng = random.Random(((int(seed) & 0xFFFFFFFF) * 100003 + 7) & 0xFFFFFFFF)
    world_pools = [MOTIF_POOLS[name] for name in WORLD_POOLS]
    spine = tuple(rng.choice(world_pools[i % len(world_pools)]) for i in range(2))
    beat_pools = [MOTIF_POOLS[name] for name in BEAT_POOLS]
    clips: dict[int, tuple] = {}
    recent: list[str] = []  # short no-repeat memory for beats
    for i in range(1, count + 1):
        pool = beat_pools[i % len(beat_pools)]
        candidate = rng.choice(pool)
        for _attempt in range(10):
            if candidate not in recent:
                break
            candidate = rng.choice(pool)
        clips[i] = (candidate,)
        recent.append(candidate)
        del recent[:-4]  # keep only a short memory so pools are not exhausted
    return {"world": spine, "clips": clips}


def render_motif_hints_block(hints) -> str:
    """Render sampled motif hints as the user-message 'ingredients' block.

    Understands the v1.1.6 shape ({"world": (..), "clips": {i: (..)}}) and
    degrades gracefully to per-clip lines for any legacy {i: (..)} dict.
    """
    if isinstance(hints, dict) and "world" in hints and "clips" in hints:
        world = "; ".join(str(h) for h in hints.get("world") or ())
        lines = [f"world spine (present and evolving in EVERY clip): {world}"]
        for i in sorted(hints.get("clips") or {}, key=lambda k: int(k)):
            beat = "; ".join(str(h) for h in hints["clips"].get(i) or ())
            lines.append(f"clip_{int(i)} beat (the change this clip contributes): {beat}")
        return "\n".join(lines)
    # legacy/foreign shape: plain per-clip lines
    lines = []
    for i in sorted((hints or {}).keys(), key=lambda k: int(k)):
        joined = "; ".join(str(h) for h in hints[i])
        lines.append(f"clip_{int(i)}: {joined}")
    return "\n".join(lines)


CREATIVE_STANDARDS_HEADER = "CREATIVE WRITING STANDARDS (fork v1.1.6)"

CREATIVE_STANDARDS_TEXT = """\
CREATIVE WRITING STANDARDS (fork v1.1.6 — applies to the GLOBAL and CLIPS prose; the output contract above always wins when they conflict):
- CONTINUITY FIRST: the sequence is one continuous evolving scene. Every clip continues the state left by the previous clip; variety comes from WHAT happens next (new concrete events, textures, micro-details), never from resetting the world, the light or the cast. The contract's continuity verbs (continues, gradually, reaches, begins, transitions into...) are welcome — vary the sentences around them, not the law itself.
- Pitch, don't fill in a form. Every clip carries ONE concrete, memorable visual idea — an image someone could describe to another person afterwards. If a clip reads like a template ("the subject continues moving..."), rewrite it before emitting it.
- Specific nouns beat generic adjectives: "grease-stained overalls", "rain beading on a chrome handrail", "a moth circling the lamp" — never "interesting clothes", "nice atmosphere", "beautiful surroundings".
- Anchor every clip physically: materials, how light behaves (hard or soft, its direction and colour), micro-events (a drip, a tremor, a glance, fabric catching air), and one sensory contrast per clip (temperature, scale, texture, stillness against motion).
- Vary the PROSE, not the world: never open two clips with the same sentence pattern, and let the dominant imagery progress step by step — each clip advances the same evolving atmosphere one clear stage further (dusk deepens, rain thickens, the crowd grows).
- Vary the verbs of the material world: spills, coils, snaps, drifts, blooms, fractures, surges, settles — concrete change verbs that carry the SAME scene forward.
- Dare inside the idea: one image per plan that daily life cannot offer (a reflection showing another time, the subject doubled in glass, smoke sinking like liquid), one transformation of state (wet to dry, calm to storm, empty to crowded) developed ACROSS several clips, one moment of held tension and one of release."""

CREATIVE_EXPRESSIVE_ADDENDUM = """\

EXPRESSIVE MODE (in addition to the standards above):
- Invent boldly: surreal but internally coherent imagery, sensory paradoxes, and a micro-story arc across the sequence — each clip escalates or answers the previous one (cause to effect, question to answer, tension to release). The arc is the through-line: bold images must still belong to one evolving world.
- The suggested visual motifs in the user message are ingredients, not a checklist: the WORLD SPINE must be present and developing in every clip; each clip's beat ingredient advances the scene one stage. Weave them invisibly into the prose; never quote them verbatim, never enumerate them.
- CAMERAS section: neighbouring cards should differ in at least two fields (shot_size, movement, rig, ...) — the framing changes while the scene continues. Exact repetition is justified only as a deliberate rhythm (a strict A-B-A pattern)."""

# Injected into the EN->RU translate system prompt when creative_boost is on:
# the translation must not flatten the freshly won vividness back to neutral
# bureaucratic Russian.
TRANSLATE_VIVID_NOTE = (
    "ХУДОЖЕСТВЕННАЯ ТОЧНОСТЬ (v1.1.5+): сохрани всю образность и конкретику "
    "оригинала — материалы, поведение света, микрособытия, смелые образы — "
    "и связность повествования между клипами (глаголы продолжения, "
    "наследование состояния сцены). Не выхолащивай текст до нейтрального "
    "пересказа: яркий английский должен остаться ярким русским. Правила "
    "сохранения разметки выше имеют абсолютный приоритет."
)


def render_creative_standards(level: str) -> str:
    """The system-prompt addendum for a creative level ('' for off)."""
    normalized, _warning = normalize_creative_boost(level)
    if normalized == "off":
        return ""
    if normalized == "standard":
        return CREATIVE_STANDARDS_TEXT
    return CREATIVE_STANDARDS_TEXT + CREATIVE_EXPRESSIVE_ADDENDUM
