from typing import List
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips


def extract_last_frame(video_path: str, output_image_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not extract last frame from {video_path}")
    cv2.imwrite(output_image_path, frame)
    return output_image_path


def merge_videos(paths: List[str], output_path: str) -> str:
    clips = [VideoFileClip(p) for p in paths]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, codec="libx264", audio=False, logger=None)
    for c in clips:
        c.close()
    final.close()
    return output_path
