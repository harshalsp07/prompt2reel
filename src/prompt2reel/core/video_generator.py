from typing import Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image


class WanVideoGenerator:
    """Low-VRAM oriented wrapper for Wan pipelines with graceful fallback."""

    def __init__(self, model_id: str, device: str = "cuda"):
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing("max")
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
        if hasattr(self.pipe, "enable_sequential_cpu_offload") and device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(device)

    def generate(
        self,
        prompt: str,
        output_path: str,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        conditioning_image_path: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        generator = torch.Generator(device="cpu").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if conditioning_image_path:
            kwargs["conditioning_image"] = Image.open(conditioning_image_path).convert("RGB")

        result = self.pipe(**kwargs)
        frames = getattr(result, "frames", None)
        if frames is None:
            raise RuntimeError("Pipeline did not return frames. Check Wan model API compatibility.")

        # defer import to avoid moviepy startup overhead unless needed
        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip([frame for frame in frames], fps=fps)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        return output_path
