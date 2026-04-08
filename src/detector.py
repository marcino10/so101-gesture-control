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

class ActivationSequenceTracker:
    def __init__(self):
        import time
        self.state_sequence = []
        self.last_state = None
        self.last_transition_time = time.time()
        
    def update(self, is_open, is_closed):
        import time
        current_state = None
        if is_open:
            current_state = "open"
        elif is_closed:
            current_state = "closed"
            
        if current_state and current_state != self.last_state:
            now = time.time()
            if now - self.last_transition_time > 0.15: # debounce short flickering
                self.state_sequence.append(current_state)
                self.last_state = current_state
                self.last_transition_time = now
                
                # Keep sliding window of 3 states
                if len(self.state_sequence) > 3:
                    self.state_sequence.pop(0)

    def is_triggered(self):
        return self.state_sequence == ["open", "closed", "open"]

    def reset(self):
        import time
        self.state_sequence = []
        self.last_state = None
        self.last_transition_time = time.time()

class HandGestureDetector:
    def __init__(self, model_path="hand_landmarker.task"):
        download_model_if_missing(model_path)
        
        self.latest_result = None
        def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result = result

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=result_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.activation_tracker = ActivationSequenceTracker()

    def is_hand_closed(self, hand_world_landmarks):
        """Checks if a hand forms a fist using 3D world distances in meters."""
        wrist = hand_world_landmarks[0]
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        closed_count = 0
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
            tip = hand_world_landmarks[tip_idx]
            mcp = hand_world_landmarks[mcp_idx]
            tip_d = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            mcp_d = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y) # We could include z, but 2D is robust for fisting
            # If the tip is closer to the wrist than the knuckle, it's curled.
            if tip_d < mcp_d + 0.01:
                closed_count += 1
        return closed_count >= 3

    def is_hand_open(self, hand_world_landmarks):
        """Checks if hand is fully spread open."""
        wrist = hand_world_landmarks[0]
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        open_count = 0
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
            tip = hand_world_landmarks[tip_idx]
            mcp = hand_world_landmarks[mcp_idx]
            tip_d = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            mcp_d = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
            # Tip should be significantly further out than knuckle (larger threshold for wider spread)
            if tip_d > mcp_d + 0.05: 
                open_count += 1
        return open_count >= 4

    def check_activation_sequence(self, hand_world_landmarks):
        """Updates internal sequence tracker and returns True if 'open-close-open'."""
        is_open = self.is_hand_open(hand_world_landmarks)
        is_closed = self.is_hand_closed(hand_world_landmarks)
        self.activation_tracker.update(is_open, is_closed)
        
        if self.activation_tracker.is_triggered():
            self.activation_tracker.reset()
            return True
        return False

    def extract_full_pose(self, hand_world_landmarks):
        """Returns a stable dictionary of all 3D joints."""
        # MediaPipe world landmarks are in meters, origin at the wrist (0).
        pose = {}
        for idx, lm in enumerate(hand_world_landmarks):
            pose[f"joint_{idx}"] = {"x": lm.x, "y": lm.y, "z": lm.z}
        return pose

    def process_async(self, rgb_frame, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)

class ArmPoseDetector:
    def __init__(self, model_path="pose_landmarker_full.task"):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        if not os.path.exists(model_path):
            print(f"Downloading {model_path} from {url}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading the pose model: {e}")
                raise e

        self.latest_result = None
        def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result = result
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
    def process_async(self, rgb_frame, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)
