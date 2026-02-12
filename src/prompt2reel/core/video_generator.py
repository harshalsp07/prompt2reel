from typing import Optional


class WanVideoGenerator:
    """Low-VRAM oriented wrapper for Wan pipelines with lazy loading and clear auth errors."""

    def __init__(self, model_id: str, device: str = "cuda", hf_token: str = ""):
        self.model_id = model_id
        self.device = device
        self.hf_token = hf_token
        self.pipe = None

    def _ensure_loaded(self) -> None:
        if self.pipe is not None:
            return
        if not self.model_id:
            raise ValueError(
                "WAN_MODEL_ID is empty. Set WAN_MODEL_ID to a valid text-to-video model repo id "
                "(and HF_TOKEN if repo is gated/private)."
            )

        import torch
        from diffusers import DiffusionPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                token=self.hf_token or None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load WAN model. Common causes: invalid WAN_MODEL_ID, missing HF_TOKEN, "
                "or no access to a gated repository."
            ) from exc

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing("max")
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
        if hasattr(self.pipe, "enable_sequential_cpu_offload") and self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.device)

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
        self._ensure_loaded()

        import torch
        from PIL import Image
        from moviepy.editor import ImageSequenceClip

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

        clip = ImageSequenceClip([frame for frame in frames], fps=fps)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        return output_path
