import os
import urllib.request
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model_if_missing(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Downloading {model_path} from {url}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise e

def is_fist(hand_landmarks, w, h):
    """Checks if a hand forms a fist using distance to wrist."""
    wrist = hand_landmarks[0]
    wx, wy = wrist.x * w, wrist.y * h
    fingers = [(8, 5), (12, 9), (16, 13), (20, 17)]
    curled_fingers = 0
    for tip_idx, mcp_idx in fingers:
        tip = hand_landmarks[tip_idx]
        mcp = hand_landmarks[mcp_idx]
        tip_d = math.hypot((tip.x * w) - wx, (tip.y * h) - wy)
        mcp_d = math.hypot((mcp.x * w) - wx, (mcp.y * h) - wy)
        if tip_d < mcp_d + 15:
            curled_fingers += 1
    return curled_fingers >= 3

class HandGestureDetector:
    def __init__(self, model_path="hand_landmarker.task"):
        download_model_if_missing(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)
