import traceback
import inspect
import gradio as gr

from prompt2reel.config.settings import DEFAULT_SETTINGS
from prompt2reel.pipelines.short_pipeline import ShortVideoPipeline


APPLE_CINEMATIC_CSS = """
body, .gradio-container {
  background: radial-gradient(circle at 20% 20%, #1f2937 0%, #0b0f19 45%, #05070c 100%) !important;
  color: #f5f5f7 !important;
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
}
#glass {
  backdrop-filter: blur(18px);
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 20px;
  box-shadow: 0 20px 80px rgba(0,0,0,0.45);
  padding: 18px;
}
button.primary {
  background: linear-gradient(120deg, #4f46e5, #06b6d4) !important;
  border: none !important;
}
"""


def _validate_settings() -> None:
    if not DEFAULT_SETTINGS.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required.")
    if not DEFAULT_SETTINGS.wan_model_id:
        raise ValueError(
            "WAN_MODEL_ID is required. Set a valid Hugging Face repo id for your WAN model. "
            "If it is gated/private, set HF_TOKEN too."
        )


def build_pipeline() -> ShortVideoPipeline:
    _validate_settings()
    return ShortVideoPipeline(DEFAULT_SETTINGS)


def run_pipeline(idea: str):
    try:
        pipeline = build_pipeline()
        final_video, prompt_dump = pipeline.run(idea)
        preview = "\n\n".join([f"{k}:\n{v}" for k, v in prompt_dump.items()])
        return final_video, preview, "✅ Generation completed"
    except Exception as exc:
        tb = traceback.format_exc(limit=2)
        return None, "", f"❌ {exc}\n\n{tb}"


def launch() -> None:
    blocks_kwargs = {}
    blocks_sig = inspect.signature(gr.Blocks)
    if "theme" in blocks_sig.parameters:
        blocks_kwargs["theme"] = gr.themes.Soft()
    if "css" in blocks_sig.parameters:
        blocks_kwargs["css"] = APPLE_CINEMATIC_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown("""
        # 🎬 Prompt2Reel — Cinematic Short Generator
        Turn one idea into a continuity-preserving 20–24s short with Gemini + Wan.

        **Before generating**, set:
        - `GEMINI_API_KEY`
        - `WAN_MODEL_ID` (valid, accessible model repo)
        - `HF_TOKEN` (if model is gated/private)
        """)

        with gr.Column(elem_id="glass"):
            idea = gr.Textbox(
                label="Your Story Idea",
                lines=3,
                placeholder="A lone astronaut discovers a neon forest on a frozen moon...",
            )
            btn = gr.Button("Generate Short", variant="primary")

        status = gr.Textbox(label="Status", lines=4)
        output_video = gr.Video(label="Final Short")
        debug_prompts = gr.Textbox(label="Generated Prompt Plan", lines=12)

        btn.click(fn=run_pipeline, inputs=idea, outputs=[output_video, debug_prompts, status])

    launch_kwargs = {"share": True}
    launch_sig = inspect.signature(demo.launch)
    if "theme" in launch_sig.parameters and "theme" not in blocks_kwargs:
        launch_kwargs["theme"] = gr.themes.Soft()
    if "css" in launch_sig.parameters and "css" not in blocks_kwargs:
        launch_kwargs["css"] = APPLE_CINEMATIC_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch()
