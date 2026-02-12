import json
from typing import Dict

import google.generativeai as genai


SYSTEM_TEMPLATE = """
You are a cinematic short-video planner.
Break idea into 3 sequential prompts for ~8s each.

Rules:
- Keep character identity, outfit, setting, and lighting continuity.
- Use specific camera movement and mood.
- Make part2/part3 naturally continue from previous parts.
- Keep prompts diffusion-friendly and visual.

Return strict JSON only:
{
  "character_memory": "short identity + style anchor",
  "world_memory": "short setting + lighting anchor",
  "part1": "...",
  "part2": "...",
  "part3": "..."
}

Idea:
{idea}
"""


class PromptPlanner:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, idea: str) -> Dict[str, str]:
        response = self.model.generate_content(SYSTEM_TEMPLATE.format(idea=idea))
        text = response.text
        start = text.find("{")
        end = text.rfind("}") + 1
        payload = json.loads(text[start:end])
        required = ["part1", "part2", "part3", "character_memory", "world_memory"]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"Planner missing keys: {missing}")
        return payload
