from prompt2reel.core.story_memory import StoryMemory


def test_top_k_retrieves_related_memory():
    memory = StoryMemory(dims=64)
    memory.add("hero wearing red coat")
    memory.add("city skyline with neon")

    ranked = memory.top_k("hero walking", k=1)
    assert ranked[0][0] == "hero wearing red coat"


def test_inject_context_adds_anchor_block():
    memory = StoryMemory(dims=64)
    memory.add("night blue lighting")

    enriched = memory.inject_context("camera tracks at night", k=1)
    assert "Continuity anchors" in enriched
    assert "night blue lighting" in enriched
