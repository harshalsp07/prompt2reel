# Prompt2Reel

Production-ready, Colab-friendly pipeline for generating a 20–24s short video from one idea:

1. Gemini generates 3 scene prompts.
2. Wan 2.2 renders 3 sequential clips (8s each).
3. Last frame of each clip conditions the next.
4. Story-memory embeddings enforce continuity.
5. Final clips are stitched into one short.
6. Apple-style Gradio UI runs directly in Google Colab.

## Highlights

- **Low-VRAM mode**: fp16/bf16, VAE slicing/tiling, attention slicing, xFormers if available, sequential CPU offload.
- **Story consistency memory embedding**: lightweight semantic memory store + retrieval injected into later prompts.
- **Production repo layout** with modular services and typed config.
- **Colab-ready cinematic UI** with progress feedback and generated prompt preview.

## Quickstart (Google Colab)

```bash
!git clone <your-repo-url>
%cd prompt2reel
!bash scripts/colab_setup.sh
!pip install -e .
!python -m prompt2reel.ui.colab_app
```

Then open the Gradio public URL shown in output.

## Environment Variables

Set in Colab before launch:

```python
import os
os.environ["GEMINI_API_KEY"] = "your_key"
```

Optional:

- `WAN_MODEL_ID` (default: `Wan-AI/Wan2.2-T2V`)
- `DEVICE` (default: `cuda`)
- `OUTPUT_DIR` (default: `outputs`)

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

## Notes

- Wan 2.2 APIs vary by release. `video_generator.py` is written to degrade gracefully if a specific argument isn't available.
- For minimal VRAM, prefer 480p/24fps or lower and 48–64 frames/clip.
