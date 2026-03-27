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

class HandGestureDetector:
    def __init__(self, model_path="hand_landmarker.task"):
        download_model_if_missing(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def get_extended_fingers(self, hand_landmarks, width, height):
        """Returns a list of finger names that are currently extended (not curled)."""
        wrist = hand_landmarks[0]
        wx, wy = wrist.x * width, wrist.y * height
        
        finger_tips = {
            "Index": (8, 5),
            "Middle": (12, 9),
            "Ring": (16, 13),
            "Pinky": (20, 17)
        }
        
        extended = []
        for name, (tip_idx, mcp_idx) in finger_tips.items():
            tip = hand_landmarks[tip_idx]
            mcp = hand_landmarks[mcp_idx]
            tip_d = math.hypot((tip.x * width) - wx, (tip.y * height) - wy)
            mcp_d = math.hypot((mcp.x * width) - wx, (mcp.y * height) - wy)
            if tip_d > mcp_d + 15:
                extended.append(name)
        return extended

    def is_fist(self, hand_landmarks, width, height):
        """Checks if a hand forms a fist using count of extended fingers."""
        extended = self.get_extended_fingers(hand_landmarks, width, height)
        return len(extended) == 0

    def get_fingertips(self, hand_landmarks, width, height):
        """Returns a dictionary of fingertip pixel coordinates."""
        fingertip_indices = {
            "Thumb": 4,
            "Index": 8,
            "Middle": 12,
            "Ring": 16,
            "Pinky": 20
        }
        coords = {}
        for name, idx in fingertip_indices.items():
            lm = hand_landmarks[idx]
            coords[name] = (int(lm.x * width), int(lm.y * height))
        return coords

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)
