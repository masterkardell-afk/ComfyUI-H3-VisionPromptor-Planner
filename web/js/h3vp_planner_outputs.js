/**
 * H3 Vision Promptor (Planner fork) — socket visibility for the two modes.
 *
 * Server-side the node always has 9 outputs in a fixed order:
 *   0 prompt, 1 vision_context, 2 debug,
 *   3 planner_prompt, 4 planner_prompt_ru, 5 global_prompt,
 *   6 clip_durations, 7 camera_params, 8 reference_images_status
 *
 * ComfyUI wires outputs BY INDEX (the frontend sends [node_id, slot_index]
 * to the backend), so only TAIL slots may ever be removed — removing a
 * HEAD slot would silently re-map every later wire to a wrong value.
 * Therefore:
 *   - original mode (emit_planner_outputs=false): the 6 planner outputs
 *     (tail, indices 3..8) are removed — the node looks exactly like the
 *     upstream original.
 *   - planner mode (true): all 9 slots stay; the two dead legacy sockets
 *     (prompt / vision_context — they return "") are greyed out and
 *     labelled "(off)". debug stays live in both modes.
 */
import { app } from "../../scripts/app.js";

const NODE_TYPE = "H3VisionPromptor";
const WIDGET_NAME = "emit_planner_outputs";
const LEGACY_DEAD = ["prompt", "vision_context"];
const PLANNER_OUTPUTS = [
  "planner_prompt",
  "planner_prompt_ru",
  "global_prompt",
  "clip_durations",
  "camera_params",
  "reference_images_status",
];
const DIM_COLOR = "#666666";
const OFF_SUFFIX = " (off)";

function isPlannerMode(node) {
  const w = node.widgets?.find((w) => w.name === WIDGET_NAME);
  return !!(w && (w.value === true || w.value === "true" || w.value === 1));
}

function rememberOriginal(node, output) {
  node.h3vp_orig ??= {};
  if (!(output.name in node.h3vp_orig)) {
    node.h3vp_orig[output.name] = {
      color_off: output.color_off,
      color_on: output.color_on,
      label: output.label,
    };
  }
}

function restoreSlotLook(node, output) {
  const orig = node.h3vp_orig?.[output.name];
  if (!orig) return;
  output.color_off = orig.color_off;
  output.color_on = orig.color_on;
  output.label = orig.label;
  delete node.h3vp_orig[output.name];
}

function dimSlotLook(node, output) {
  rememberOriginal(node, output);
  output.color_off = DIM_COLOR;
  output.color_on = DIM_COLOR;
  if (!output.label || !output.label.endsWith(OFF_SUFFIX)) {
    output.label = (output.label || output.name) + OFF_SUFFIX;
  }
}

function applyVisibility(node) {
  if (!node || node.type !== NODE_TYPE) return;
  const planner = isPlannerMode(node);

  // 1) Tail planner outputs: remove in original mode, (re-)add in planner mode.
  for (let i = node.outputs.length - 1; i >= 0; i--) {
    const output = node.outputs[i];
    if (output && PLANNER_OUTPUTS.includes(output.name)) {
      if (!planner) {
        if (output.links && output.links.length) {
          console.warn(
            "[H3VisionPromptor-Planner] removing wired planner output '" +
              output.name +
              "' (it is empty in original mode)"
          );
        }
        node.removeOutput(i);
      }
    }
  }
  if (planner) {
    const have = new Set(node.outputs.map((o) => o?.name));
    for (const name of PLANNER_OUTPUTS) {
      if (!have.has(name)) {
        node.addOutput(name, "STRING");
      }
    }
  }

  // 2) Dead legacy sockets (head slots — never removed, only dimmed/restored).
  for (const output of node.outputs ?? []) {
    if (!output) continue;
    if (planner && LEGACY_DEAD.includes(output.name)) {
      dimSlotLook(node, output);
    } else if (node.h3vp_orig?.[output.name] && !LEGACY_DEAD.includes(output.name)) {
      restoreSlotLook(node, output);
    } else if (!planner && LEGACY_DEAD.includes(output.name)) {
      restoreSlotLook(node, output);
    }
  }

  node.setDirtyCanvas?.(true, true);
}

let wrapped = new WeakSet();
function hookWidget(node) {
  const w = node.widgets?.find((w) => w.name === WIDGET_NAME);
  if (!w || wrapped.has(w)) return;
  wrapped.add(w);
  const prev = w.callback;
  w.callback = function () {
    const result = prev?.apply(this, arguments);
    applyVisibility(node);
    return result;
  };
}

app.registerExtension({
  name: "H3VisionPromptor.PlannerOutputs",
  nodeCreated(node) {
    if (node?.type !== NODE_TYPE) return;
    // Widgets may not exist yet at nodeCreated; retry once on the next tick.
    hookWidget(node);
    setTimeout(() => {
      hookWidget(node);
      applyVisibility(node);
    }, 0);
  },
  loadedGraphNode(node) {
    if (node?.type !== NODE_TYPE) return;
    hookWidget(node);
    applyVisibility(node);
  },
});
