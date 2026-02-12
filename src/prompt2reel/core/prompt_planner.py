import json
from typing import Dict


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
        self.model_name = model_name
        self._mode = "legacy"
        self.client = None
        self.model = None

        try:
            from google import genai  # type: ignore

            self.client = genai.Client(api_key=api_key)
            self._mode = "genai"
        except Exception:
            # Backward compatibility for environments that still only have google-generativeai.
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self._mode = "legacy"

    def generate(self, idea: str) -> Dict[str, str]:
        prompt = SYSTEM_TEMPLATE.format(idea=idea)

        if self._mode == "genai":
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            text = response.text or ""
        else:
            response = self.model.generate_content(prompt)
            text = response.text

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Gemini response did not contain JSON payload.")

        payload = json.loads(text[start:end])
        required = ["part1", "part2", "part3", "character_memory", "world_memory"]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"Planner missing keys: {missing}")
        return payload
