# Google Colab Run Guide

## 1) Clone and setup
```python
!git clone <your-repo-url>
%cd prompt2reel
!bash scripts/colab_setup.sh
!pip install -e .
```

## 2) Set API key
```python
import os
os.environ["GEMINI_API_KEY"] = "YOUR_KEY"
```

## 3) Launch cinematic UI
```python
!python -m prompt2reel.ui.colab_app
```

## 4) Low-VRAM tips
```python
import os
os.environ["FRAMES_PER_CLIP"] = "48"
os.environ["FPS"] = "8"
os.environ["GUIDANCE_SCALE"] = "5.5"
os.environ["RESOLUTION"] = "640x360"
```
