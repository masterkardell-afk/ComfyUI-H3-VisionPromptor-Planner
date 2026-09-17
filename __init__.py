"""ComfyUI-H3-VisionPromptor-Planner — fork of benjiyaya/ComfyUI-H3-VisionPromptor.

Adds the LongMedia Planner mode to H3VisionPromptor: with
emit_planner_outputs=true the node emits planner_prompt / planner_prompt_ru /
global_prompt / clip_durations / camera_params / reference_images_status
(STRING) for the ComfyUI-MiniMax-H3-LongMedia Planner + Cameras nodes, and
disables its original prompt/vision_context outputs. With
emit_planner_outputs=false it behaves byte-identically to the original node.

Registers the V3 extension when a recent ComfyUI (comfy_api.latest + native
text generation) is available; otherwise prints an actionable message instead
of crashing the whole custom-node load.
"""

# Frontend assets (socket visibility for the two modes). The classic
# WEB_DIRECTORY attribute is honored by ComfyUI's custom-node loader for both
# V1 and V3 packages; [tool.comfy] web in pyproject.toml is the declarative
# twin.
WEB_DIRECTORY = "./web"

try:
    from comfy_api.latest import ComfyExtension, io  # noqa: F401
    from .py.nodes import H3VisionPromptor, H3VisionAnalyzer

    class H3VisionPromptorExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3VisionPromptor, H3VisionAnalyzer]

    async def comfy_entrypoint():
        return H3VisionPromptorExtension()

except ImportError as e:  # old ComfyUI — don't crash the whole custom-node load
    print(f"[ComfyUI-H3-VisionPromptor-Planner] requires a recent ComfyUI "
          f"(comfy_api.latest + native text generation). Error: {e}")
