import pytest

from prompt2reel.core.video_generator import WanVideoGenerator


def test_missing_model_id_raises_clear_error():
    generator = WanVideoGenerator(model_id="", device="cpu")
    with pytest.raises(ValueError) as exc:
        generator._ensure_loaded()
    assert "WAN_MODEL_ID is empty" in str(exc.value)
