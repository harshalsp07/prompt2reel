from dataclasses import dataclass
import os


@dataclass
class AppSettings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    # Use an explicit placeholder to force user choice; many WAN repos are gated/private.
    wan_model_id: str = os.getenv("WAN_MODEL_ID", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    device: str = os.getenv("DEVICE", "cuda")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    fps: int = int(os.getenv("FPS", "8"))
    frames_per_clip: int = int(os.getenv("FRAMES_PER_CLIP", "64"))
    guidance_scale: float = float(os.getenv("GUIDANCE_SCALE", "6.5"))
    resolution: str = os.getenv("RESOLUTION", "848x480")
    seed: int = int(os.getenv("SEED", "42"))


DEFAULT_SETTINGS = AppSettings()
