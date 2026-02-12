import os
from typing import Dict, Tuple

from prompt2reel.config.settings import AppSettings
from prompt2reel.core.prompt_planner import PromptPlanner
from prompt2reel.core.story_memory import StoryMemory
from prompt2reel.core.video_generator import WanVideoGenerator
from prompt2reel.utils.video_ops import extract_last_frame, merge_videos


class ShortVideoPipeline:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        os.makedirs(self.settings.output_dir, exist_ok=True)
        self.planner = PromptPlanner(settings.gemini_api_key, settings.gemini_model)
        self.memory = StoryMemory()
        self.video = WanVideoGenerator(settings.wan_model_id, settings.device, settings.hf_token)

    def _paths(self) -> Dict[str, str]:
        od = self.settings.output_dir
        return {
            "v1": os.path.join(od, "part1.mp4"),
            "v2": os.path.join(od, "part2.mp4"),
            "v3": os.path.join(od, "part3.mp4"),
            "f1": os.path.join(od, "frame1.png"),
            "f2": os.path.join(od, "frame2.png"),
            "final": os.path.join(od, "final_short.mp4"),
        }

    def run(self, idea: str) -> Tuple[str, Dict[str, str]]:
        prompts = self.planner.generate(idea)
        self.memory.add(prompts["character_memory"])
        self.memory.add(prompts["world_memory"])

        p = self._paths()
        part1 = self.memory.inject_context(prompts["part1"])
        self.video.generate(part1, p["v1"], self.settings.frames_per_clip, self.settings.fps, self.settings.guidance_scale, seed=self.settings.seed)
        extract_last_frame(p["v1"], p["f1"])

        part2 = self.memory.inject_context(prompts["part2"])
        self.video.generate(part2, p["v2"], self.settings.frames_per_clip, self.settings.fps, self.settings.guidance_scale, conditioning_image_path=p["f1"], seed=self.settings.seed + 1)
        extract_last_frame(p["v2"], p["f2"])

        part3 = self.memory.inject_context(prompts["part3"])
        self.video.generate(part3, p["v3"], self.settings.frames_per_clip, self.settings.fps, self.settings.guidance_scale, conditioning_image_path=p["f2"], seed=self.settings.seed + 2)

        final = merge_videos([p["v1"], p["v2"], p["v3"]], p["final"])
        return final, {"part1": part1, "part2": part2, "part3": part3}
