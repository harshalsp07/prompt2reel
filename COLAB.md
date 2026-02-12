# Google Colab Run Guide

## 1) Clone and setup
```python
!git clone <your-repo-url>
%cd prompt2reel
!bash scripts/colab_setup.sh
!pip install -e .
```

## 2) Set API keys and model access
```python
import os
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_KEY"
os.environ["WAN_MODEL_ID"] = "YOUR_ACCESSIBLE_VIDEO_MODEL_ID_OR_LOCAL_PATH"
# required if WAN model is gated/private on Hugging Face
os.environ["HF_TOKEN"] = "hf_xxx"
```


## 3) (Optional but recommended) Download WAN model locally
```python
import os
os.environ["WAN_MODEL_ID"] = "YOUR_ACCESSIBLE_VIDEO_MODEL_ID"
os.environ["HF_TOKEN"] = "hf_xxx"  # if model is gated/private
!bash scripts/setup_wan22.sh

# then use local path for runtime stability
os.environ["WAN_MODEL_ID"] = "models/wan22"
```

## 4) Launch cinematic UI
```python
!python -m prompt2reel.ui.colab_app
```

## 5) Low-VRAM tips
```python
import os
os.environ["FRAMES_PER_CLIP"] = "48"
os.environ["FPS"] = "8"
os.environ["GUIDANCE_SCALE"] = "5.5"
os.environ["RESOLUTION"] = "640x360"
```
