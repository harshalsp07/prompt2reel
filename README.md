# Prompt2Reel

Production-ready, Colab-friendly pipeline for generating a 20–24s short video from one idea:

1. Gemini generates 3 scene prompts.
2. Wan-compatible model renders 3 sequential clips (8s each).
3. Last frame of each clip conditions the next.
4. Story-memory embeddings enforce continuity.
5. Final clips are stitched into one short.
6. Apple-style Gradio UI runs directly in Google Colab.

## Why your previous run failed

The earlier default `WAN_MODEL_ID="Wan-AI/Wan2.2-T2V"` is not publicly accessible in many environments, which causes Hugging Face `401/RepositoryNotFound` errors. This repo now:

- requires explicit `WAN_MODEL_ID` configuration,
- accepts `HF_TOKEN` for gated/private models,
- loads the video model lazily only when generation starts,
- shows user-friendly error messages in the UI instead of crashing the whole app.

## Highlights

- **Low-VRAM mode**: fp16/bf16, VAE slicing/tiling, attention slicing, xFormers if available, sequential CPU offload.
- **Story consistency memory embedding**: lightweight semantic memory store + retrieval injected into later prompts.
- **Production repo layout** with modular services and typed config.
- **Colab-ready cinematic UI** with progress feedback and generated prompt preview.
- **Gemini SDK compatibility**: uses modern `google.genai` when available and falls back to `google.generativeai`.

## Quickstart (Google Colab)

```bash
!git clone <your-repo-url>
%cd prompt2reel
!bash scripts/colab_setup.sh
!pip install -e .
```

Set required environment variables:

```python
import os
os.environ["GEMINI_API_KEY"] = "your_key"
os.environ["WAN_MODEL_ID"] = "<your-accessible-video-model-repo-or-local-path>"
# If model is gated/private on Hugging Face:
os.environ["HF_TOKEN"] = "hf_xxx"
```

Launch app:

```bash
!python -m prompt2reel.ui.colab_app
```


### Optional: pre-download WAN model (recommended in Colab)

```bash
# If model is gated/private, set HF_TOKEN first
!export WAN_MODEL_ID="<your-model-repo>"
!export HF_TOKEN="hf_xxx"
!bash scripts/setup_wan22.sh

# Then point app to local downloaded files
import os
os.environ["WAN_MODEL_ID"] = "models/wan22"
```

## Optional Environment Variables

- `GEMINI_MODEL` (default: `gemini-1.5-pro`)
- `DEVICE` (default: `cuda`)
- `OUTPUT_DIR` (default: `outputs`)
- `FRAMES_PER_CLIP` (default: `64`)
- `FPS` (default: `8`)
- `GUIDANCE_SCALE` (default: `6.5`)

## Project Structure

```text
src/prompt2reel/
  config/         # Settings dataclasses
  core/           # Prompt generation, memory embeddings, video generation
  pipelines/      # End-to-end orchestration
  ui/             # Gradio Apple-style cinematic dashboard
  utils/          # ffmpeg and frame extraction helpers
scripts/
  colab_setup.sh  # Colab dependency setup
tests/
  test_memory.py  # Story memory behavior smoke tests
```
