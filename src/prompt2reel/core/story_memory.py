from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple


@dataclass
class MemoryEntry:
    text: str
    embedding: List[float]


class StoryMemory:
    """Tiny dependency-free semantic memory for continuity anchors."""

    def __init__(self, dims: int = 128):
        self.dims = dims
        self.entries: List[MemoryEntry] = []

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dims
        tokens = [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split()]
        for tok in tokens:
            if not tok:
                continue
            idx = hash(tok) % self.dims
            vec[idx] += 1.0
        norm = sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def add(self, text: str) -> None:
        self.entries.append(MemoryEntry(text=text, embedding=self._embed(text)))

    def top_k(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.entries:
            return []
        q = self._embed(query)
        scored = [(entry.text, self._dot(q, entry.embedding)) for entry in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def inject_context(self, prompt: str, k: int = 3) -> str:
        memory_bits = self.top_k(prompt, k=k)
        if not memory_bits:
            return prompt
        anchors = "\n".join([f"- {txt}" for txt, _ in memory_bits])
        return f"{prompt}\n\nContinuity anchors:\n{anchors}"
