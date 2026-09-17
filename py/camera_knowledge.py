"""Camera knowledge base for the LongMedia Planner mode.

Autonomous copy of the MiniMax-H3-LongMedia Cameras node vocabulary:
every enum field of a LongMedia camera card, every allowed value, and a
short English semantic description of what each value produces. The fork
deliberately does NOT import ComfyUI-MiniMax-H3-LongMedia — this file is a
self-contained mirror so the two packs can be installed / updated
independently.

Sources mirrored (vizart-vj/ComfyUI-MiniMax-H3-LongMedia v0.6.40):
  - web/longmedia_cameras.js            (allowed value lists)
  - nodes.py  _LONGMEDIA_CAMERA_*       (per-value semantic descriptions)

If LongMedia changes its vocabulary, re-mirror this file.
"""

from __future__ import annotations

import difflib

# Ordered card keys exactly as the LongMedia Cameras card / camera_plan uses.
CAMERA_CARD_KEYS = (
    "clip_id",
    "clip_name",
    "shot_size",
    "rig",
    "camera_body",
    "lens",
    "stabilization",
    "movement",
    "speed",
    "transition_type",
    "space_relation",
    "entity_continuity",
    "transition_to_next",
)

SHOT_SIZES = {
    "Extreme Wide Shot": "extreme wide shot, subject very small within a vast environment",
    "Wide Shot": "wide shot showing the full subject and substantial environment",
    "Full Shot": "full-body shot from head to toe",
    "Cowboy Shot": "cowboy shot framed from approximately mid-thigh upward",
    "Medium Full Shot": "medium-full shot framed roughly from the knees upward",
    "Medium Shot": "medium shot framed approximately from the waist upward",
    "Medium Close-Up": "medium close-up framed from the chest or shoulders upward",
    "Close-Up": "close-up emphasizing the face and upper shoulders",
    "Extreme Close-Up": "extreme close-up emphasizing facial details or a single expressive feature",
    "Macro / Detail": "macro detail shot with very tight framing on a small subject detail",
    "Over-the-Shoulder": "over-the-shoulder framing with a foreground shoulder or silhouette",
    "Two-Shot": "two-shot composed to hold two subjects clearly in the frame",
    "POV Framing": "subjective point-of-view framing from the observer's perspective",
}

RIGS = {
    "Tripod / Locked Head": "camera mounted on a rigid tripod with a locked or controlled head",
    "Fluid Head Tripod": "camera mounted on a professional fluid-head tripod",
    "Dolly / Track": "camera mounted on a cinema dolly or linear track",
    "Slider": "camera mounted on a compact motorized slider",
    "Jib / Crane": "camera mounted on a jib or crane arm",
    "Technocrane": "camera mounted on a telescopic Technocrane",
    "Steadicam": "camera mounted on a body-worn Steadicam stabilization rig",
    "3-Axis Gimbal": "camera mounted on a motorized three-axis gimbal",
    "Shoulder Rig": "camera mounted on a shoulder rig",
    "Handheld": "camera operated handheld",
    "Vehicle Mount": "camera mounted to a moving vehicle or pursuit platform",
    "Cable Cam": "camera suspended on a cable-cam system",
    "Robot Arm · Bolt": "camera mounted on a high-speed MRMC Bolt-style cinema robot arm",
    "Robot Arm · KUKA": "camera mounted on an industrial KUKA-style motion-control robot arm",
    "Drone · Heavy-Lift Cinema": "camera carried by a heavy-lift professional cinema drone",
    "Drone · DJI Inspire 3": "camera carried by a DJI Inspire 3 professional aerial platform",
    "Drone · DJI Mavic 3 Cine": "camera carried by a DJI Mavic 3 Cine aerial platform",
    "Drone · DJI Air 3S": "camera carried by a DJI Air 3S aerial platform",
    "Drone · DJI Mini 4 Pro": "camera carried by a DJI Mini 4 Pro compact aerial platform",
    "FPV · DJI Avata 2": "camera carried by a DJI Avata 2 FPV platform",
    "FPV · Cinewhoop": "camera carried by a compact cinewhoop FPV platform",
    "FPV · Racing": "camera carried by a high-speed racing FPV platform",
    "Bodycam Mount": "camera fixed to a body-worn mount",
    "Helmet / Head Mount": "camera fixed to a head or helmet mount",
    "Static Security Mount": "camera fixed to a rigid surveillance mount",
}

CAMERA_BODIES = {
    "Cinematic Neutral": "high-end neutral digital cinema camera response",
    "ARRI Alexa 35": "ARRI Alexa 35 digital cinema camera with natural highlight roll-off and rich dynamic range",
    "ARRI Alexa Mini LF": "ARRI Alexa Mini LF large-format cinema camera with soft highlight roll-off",
    "Sony VENICE 2": "Sony VENICE 2 full-frame digital cinema camera",
    "RED V-RAPTOR XL": "RED V-RAPTOR XL high-resolution digital cinema camera",
    "RED KOMODO-X": "RED KOMODO-X compact global-shutter cinema camera",
    "Blackmagic URSA Cine 12K": "Blackmagic URSA Cine 12K digital cinema camera",
    "Sony FX3": "Sony FX3 compact full-frame cinema camera",
    "Sony FX6": "Sony FX6 documentary-oriented full-frame cinema camera",
    "Canon C400": "Canon C400 digital cinema camera",
    "Canon EOS R5 C": "Canon EOS R5 C hybrid cinema camera",
    "Canon EOS 5D Mark II": "Canon EOS 5D Mark II DSLR video camera",
    "Nikon D850": "Nikon D850 DSLR camera",
    "Sony DCR-VX1000": "Sony DCR-VX1000 MiniDV camcorder",
    "Canon XL1": "Canon XL1 MiniDV camcorder",
    "Panasonic DVX100": "Panasonic DVX100 MiniDV camcorder",
    "VHS Camcorder": "full-size consumer VHS camcorder",
    "VHS-C Camcorder": "compact VHS-C analog camcorder",
    "Sony Hi8 Handycam": "Sony Hi8 analog Handycam",
    "Super 8 Camera": "Super 8 small-gauge film camera",
    "Aaton XTR 16mm": "Aaton XTR 16mm motion-picture camera",
    "Arricam LT 35mm": "Arricam LT 35mm motion-picture camera",
    "IMAX 65mm": "IMAX 65mm large-format motion-picture camera",
    "Smartphone · Snapshot": "modern flagship smartphone in casual snapshot video mode",
    "Smartphone · Cinematic": "modern flagship smartphone in computational cinematic video mode",
    "Action Camera": "compact wide-angle action camera",
    "Broadcast ENG": "professional broadcast ENG camera",
    "CCTV Sensor": "utilitarian surveillance camera sensor",
    "Webcam": "consumer webcam imaging system",
}

LENSES = {
    "Auto / Native Lens": "natural lens choice appropriate to the selected camera body, rig and shot size",
    "Ultra-Wide 10mm": "10mm rectilinear ultra-wide lens with extreme spatial expansion",
    "Ultra-Wide 12mm": "12mm ultra-wide lens with strong environmental perspective",
    "Ultra-Wide 14mm": "14mm ultra-wide cinema lens",
    "Wide 18mm": "18mm wide-angle cinema lens",
    "Wide 21mm": "21mm wide-angle cinema lens",
    "Wide 24mm": "24mm wide-angle cinema lens",
    "Wide 28mm": "28mm moderate wide-angle lens",
    "Natural 35mm": "35mm natural wide-normal cinema lens",
    "Natural 40mm": "40mm natural perspective cinema lens",
    "Standard 50mm": "50mm standard lens with natural perspective",
    "Portrait 65mm": "65mm short-tele portrait cinema lens",
    "Portrait 85mm": "85mm portrait lens with compressed perspective and shallow depth",
    "Telephoto 100mm": "100mm telephoto lens",
    "Telephoto 135mm": "135mm telephoto lens with strong spatial compression",
    "Long Telephoto 200mm": "200mm long telephoto lens",
    "Long Telephoto 300mm": "300mm long telephoto lens with very strong compression",
    "Macro 60mm": "60mm macro lens for close detail",
    "Macro 100mm": "100mm macro lens for extreme close detail",
    "Anamorphic 28mm": "28mm anamorphic cinema lens",
    "Anamorphic 35mm": "35mm anamorphic cinema lens with horizontal field character and oval bokeh",
    "Anamorphic 50mm": "50mm anamorphic cinema lens with cinematic compression and oval bokeh",
    "Anamorphic 75mm": "75mm anamorphic cinema lens with portrait compression",
    "Vintage Spherical · Wide": "vintage wide spherical cinema lens with softer contrast and organic aberrations",
    "Vintage Spherical · Normal": "vintage normal spherical cinema lens with softer contrast and organic aberrations",
    "Vintage Spherical · Portrait": "vintage portrait spherical cinema lens with soft roll-off and organic aberrations",
    "Probe Lens": "long probe macro lens for extreme close-range moving shots",
    "Tilt-Shift": "tilt-shift lens with selective plane-of-focus control",
    "Fisheye": "fisheye lens with extreme curved ultra-wide perspective",
    "Smartphone Ultra-Wide": "smartphone computational ultra-wide lens",
    "Smartphone Wide": "smartphone computational wide lens",
    "Smartphone Tele": "smartphone computational telephoto lens",
}

STABILIZATION = {
    "Rig Native": "use the natural stabilization behavior of the selected rig",
    "Hard Locked": "mechanically locked orientation with no operator drift",
    "Fluid Controlled": "fluid controlled stabilized motion with gentle acceleration",
    "Gyro Stabilized": "strong gyroscopic stabilization with horizon control",
    "Gimbal Smooth": "motorized gimbal stabilization with polished floating motion",
    "Steadicam Organic": "Steadicam stabilization with subtle organic operator drift",
    "Handheld Controlled": "restrained handheld micro-motion",
    "Handheld Raw": "raw handheld movement with stronger natural micro-jitter",
    "FPV Stabilized": "stabilized FPV motion retaining agile flight characteristics",
    "FPV Raw": "direct FPV flight feel with stronger banking and rotation",
}

MOVEMENTS = {
    "Locked-Off / Static": "camera remains completely locked-off and static",
    "Push-In": "camera moves directly forward toward the subject",
    "Pull-Out": "camera moves directly backward away from the subject",
    "Track Forward": "camera translates forward through the scene",
    "Track Backward": "camera translates backward through the scene",
    "Track Left": "camera translates laterally to the left",
    "Track Right": "camera translates laterally to the right",
    "Pan Left": "camera rotates horizontally to the left from its position",
    "Pan Right": "camera rotates horizontally to the right from its position",
    "Tilt Up": "camera rotates vertically upward from its position",
    "Tilt Down": "camera rotates vertically downward from its position",
    "Crane Up": "camera rises vertically upward",
    "Crane Down": "camera descends vertically downward",
    "Pedestal Up": "camera body moves straight upward while preserving viewing direction",
    "Pedestal Down": "camera body moves straight downward while preserving viewing direction",
    "Arc Left": "camera moves on a partial circular arc around the subject toward the left",
    "Arc Right": "camera moves on a partial circular arc around the subject toward the right",
    "Orbit Clockwise": "camera performs a circular orbit around the subject in a clockwise direction",
    "Orbit Counterclockwise": "camera performs a circular orbit around the subject in a counterclockwise direction",
    "Full 360 Orbit Clockwise": "camera completes a full 360-degree circular fly-around around the subject clockwise",
    "Full 360 Orbit Counterclockwise": "camera completes a full 360-degree circular fly-around around the subject counterclockwise",
    "Half Orbit Clockwise": "camera performs an approximately 180-degree circular fly-around around the subject clockwise",
    "Half Orbit Counterclockwise": "camera performs an approximately 180-degree circular fly-around around the subject counterclockwise",
    "Spiral In Clockwise": "camera circles clockwise while gradually moving closer to the subject",
    "Spiral In Counterclockwise": "camera circles counterclockwise while gradually moving closer to the subject",
    "Spiral Out Clockwise": "camera circles clockwise while gradually moving farther from the subject",
    "Spiral Out Counterclockwise": "camera circles counterclockwise while gradually moving farther from the subject",
    "Diagonal Forward Left": "camera moves diagonally forward and to the left",
    "Diagonal Forward Right": "camera moves diagonally forward and to the right",
    "Diagonal Backward Left": "camera moves diagonally backward and to the left",
    "Diagonal Backward Right": "camera moves diagonally backward and to the right",
    "Rise + Push-In": "camera rises while simultaneously moving forward toward the subject",
    "Descend + Push-In": "camera descends while simultaneously moving forward toward the subject",
    "Rise + Pull-Out": "camera rises while simultaneously moving backward away from the subject",
    "Descend + Pull-Out": "camera descends while simultaneously moving backward away from the subject",
}

SPEEDS = {
    "Static": "no visible viewpoint motion",
    "Ultra Slow": "extremely slow, almost imperceptible viewpoint motion",
    "Slow": "slow and deliberate viewpoint motion",
    "Controlled": "measured, polished, controlled viewpoint motion",
    "Medium": "moderate natural viewpoint motion",
    "Fast": "fast purposeful viewpoint motion",
    "Aggressive": "aggressive high-energy viewpoint motion",
    "Variable / Ramping": "viewpoint speed changes smoothly with cinematic acceleration and deceleration",
}

# The three dicts below have internal codes (not prose) in LongMedia; short
# semantic descriptions here are written for the fork's knowledge injection.
TRANSITION_TYPES = {
    "Continuous / Same Shot": "no cut — the viewpoint flows continuously into the next clip",
    "Threshold Entry": "the next clip enters exactly when the subject crosses a physical threshold (door, arch, light edge)",
    "Occluded Hidden Cut": "the cut is hidden behind a full-frame occlusion (a passing body, wall, foliage)",
    "Hard Cut": "an explicit hard cut at the clip boundary",
}

SPACE_RELATIONS = {
    "Same Space": "the next clip continues in the exact same physical space and layout",
    "Adjacent Space": "the next clip moves to a directly adjacent, physically connected space",
    "Different Space": "the next clip moves to a different, unconnected space",
}

ENTITY_CONTINUITY = {
    "Lock Population / Layout": "all characters and objects keep their exact positions across the boundary",
    "Preserve Main Subjects": "main subjects persist; background population may evolve",
    "Allow Background Evolution": "main subjects persist; the background is free to change",
}

# field -> allowed values (with semantics)
ENUM_FIELDS = {
    "shot_size": SHOT_SIZES,
    "rig": RIGS,
    "camera_body": CAMERA_BODIES,
    "lens": LENSES,
    "stabilization": STABILIZATION,
    "movement": MOVEMENTS,
    "speed": SPEEDS,
    "transition_type": TRANSITION_TYPES,
    "space_relation": SPACE_RELATIONS,
    "entity_continuity": ENTITY_CONTINUITY,
}

# Defaults mirror LongMedia's _lm_camera_default_card().
DEFAULT_CARD = {
    "shot_size": "Medium Shot",
    "rig": "Tripod / Locked Head",
    "camera_body": "Cinematic Neutral",
    "lens": "Auto / Native Lens",
    "stabilization": "Rig Native",
    "movement": "Locked-Off / Static",
    "speed": "Static",
    "transition_type": "Continuous / Same Shot",
    "space_relation": "Same Space",
    "entity_continuity": "Lock Population / Layout",
}


def _norm(value: str) -> str:
    """Case/punctuation-insensitive normal form used for fuzzy matching."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


_LOOKUP = {field: {_norm(v): v for v in allowed} for field, allowed in ENUM_FIELDS.items()}


def resolve_enum_value(field: str, value) -> tuple[str | None, str | None]:
    """Resolve a model-emitted value against the allowed enum for a field.

    Returns (resolved_value, note). resolved_value is None when nothing
    matches. note describes what happened (exact / normalized / fuzzy / empty).
    """
    allowed = ENUM_FIELDS.get(field)
    if allowed is None:
        return None, None
    raw = str(value or "").strip()
    if not raw:
        return None, "empty"
    if raw in allowed:
        return raw, "exact"
    norm = _norm(raw)
    hit = _LOOKUP[field].get(norm)
    if hit is not None:
        return hit, f"normalized from '{raw}'"
    close = difflib.get_close_matches(norm, list(_LOOKUP[field].keys()), n=1, cutoff=0.86)
    if close:
        return _LOOKUP[field][close[0]], f"normalized from '{raw}'"
    return None, f"invalid value '{raw}'"


def normalize_card(card, index: int) -> tuple[dict, list[str]]:
    """Validate/normalize one camera card against the LongMedia vocabulary.

    Unknown fields are dropped, missing fields fall back to the LongMedia
    default, invalid values are fuzzy-matched or replaced by the default,
    clip_id is forced to 'clip-{index+1}' and transition_to_next is coerced
    to a bool. Returns (card, warnings).
    """
    warnings: list[str] = []
    if not isinstance(card, dict):
        card = {}
        warnings.append(f"camera card {index + 1} is not an object; replaced with defaults")

    out: dict = {}
    known = set(CAMERA_CARD_KEYS)
    for key in card:
        if key not in known:
            warnings.append(f"camera card {index + 1}: dropped unknown field '{key}'")

    for field, allowed in ENUM_FIELDS.items():
        if field not in card:
            out[field] = DEFAULT_CARD[field]
            warnings.append(f"camera card {index + 1}: missing '{field}', used default "
                            f"'{DEFAULT_CARD[field]}'")
            continue
        resolved, note = resolve_enum_value(field, card[field])
        if resolved is None:
            out[field] = DEFAULT_CARD[field]
            warnings.append(f"camera card {index + 1}: {note} for '{field}', "
                            f"used default '{DEFAULT_CARD[field]}'")
        else:
            out[field] = resolved
            if note not in ("exact",):
                warnings.append(f"camera card {index + 1}: '{field}' {note} -> '{resolved}'")

    ttn = card.get("transition_to_next", True)
    if isinstance(ttn, str):
        ttn = ttn.strip().lower() in ("true", "1", "yes")
    out["transition_to_next"] = bool(ttn)

    out["clip_id"] = f"clip-{index + 1}"
    out["clip_name"] = str(card.get("clip_name") or card.get("name") or "").strip()[:120]

    # Keep the key order identical to CAMERA_CARD_KEYS.
    ordered = {key: out[key] for key in CAMERA_CARD_KEYS}
    return ordered, warnings


def render_knowledge_block() -> str:
    """Render the full camera vocabulary for injection into the system prompt."""
    lines = []
    for field, allowed in ENUM_FIELDS.items():
        lines.append(f"FIELD {field} — allowed values (copy EXACTLY, including case/punctuation):")
        for value, desc in allowed.items():
            lines.append(f'  - "{value}" — {desc}')
        lines.append("")
    lines.append("FIELD transition_to_next — boolean true/false. Set false for the LAST card; "
                 "true means this clip's camera state flows into the next clip.")
    lines.append("FIELD clip_id — literal 'clip-1', 'clip-2', ... matching the clip index.")
    lines.append("FIELD clip_name — short English label (max 3 words) or \"\".")
    return "\n".join(lines)


def enum_sizes() -> dict:
    """Field -> number of allowed values (for tests / diagnostics)."""
    return {field: len(allowed) for field, allowed in ENUM_FIELDS.items()}
