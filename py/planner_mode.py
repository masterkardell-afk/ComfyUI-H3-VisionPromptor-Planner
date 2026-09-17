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
  - estimate_planner_tokens() / estimate_translate_tokens(): token budgets
    (planner mode requests its own budget, independent of the max_tokens
    widget).
"""

from __future__ import annotations

import json
import math
import re

from . import camera_knowledge

MIN_CLIPS = 2
MAX_CLIPS = 16
MAX_CLIP_SECONDS = 150.0
MIN_TOTAL_SECONDS = 8.0
MAX_TOTAL_SECONDS = 2400.0

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
# Token budgets
# ---------------------------------------------------------------------------

def _round_up64(value: int) -> int:
    return int(math.ceil(max(0, value) / 64.0)) * 64


def estimate_planner_tokens(clip_count: int) -> int:
    """Token budget for the single planner generation pass.

    ~110 tokens per clip body + ~120 tokens per camera card + global +
    section overhead + safety margin.
    """
    count = max(1, int(clip_count))
    estimate = 360 + count * 260
    return int(min(8192, max(2048, _round_up64(estimate))))


def estimate_translate_tokens(text: str) -> int:
    """Token budget for the EN->RU mirror pass (RU words tokenize heavier)."""
    words = len(str(text or "").split())
    estimate = int(words * 2.2) + 300
    return int(min(8192, max(1024, _round_up64(estimate))))
