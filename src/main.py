import cv2
import math
from detector import HandGestureDetector, is_fist
from robot_controller import RobotController

def main():
    detector = HandGestureDetector(model_path="hand_landmarker.task")
    fingertip_indices = [4, 8, 12, 16, 20]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam... Press 'q' to quit.")

    state = "IDLE"
    baseline_center = None
    baseline_box_half_size = 0
    missing_frames = 0
    locked_hand_label = None

    # Connect to the robot automatically, using a context manager
    with RobotController(port="/dev/ttyACM0") as robot:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = detector.process(rgb_frame)

            # Handle State Reset if hands disappear
            if not detection_result.hand_landmarks:
                missing_frames += 1
                if missing_frames > 20 and state == "ACTIVE":
                    print("\n>>> HAND LOST: Resetting to IDLE mode <<<")
                    state = "IDLE"
                    baseline_center = None
                    locked_hand_label = None
                
                cv2.putText(frame, f"STATE: {state}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                missing_frames = 0
                
                for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    handedness_list = detection_result.handedness
                    current_hand_label = handedness_list[hand_idx][0].category_name if (handedness_list and len(handedness_list) > hand_idx) else "Unknown"
                    
                    if state == "IDLE":
                        if is_fist(hand_landmarks, w, h):
                            center_lm = hand_landmarks[9] 
                            baseline_center = (int(center_lm.x * w), int(center_lm.y * h))
                            wrist = hand_landmarks[0]
                            hand_size = math.hypot((wrist.x * w) - baseline_center[0], (wrist.y * h) - baseline_center[1])
                            
                            baseline_box_half_size = int((hand_size * 2.05) / 2)
                            if baseline_box_half_size < 50:
                                baseline_box_half_size = 50
                                
                            state = "ACTIVE"
                            locked_hand_label = current_hand_label
                            print(f"\n>>> FIST DETECTED: Locked onto {locked_hand_label} Hand. <<<")
                    
                    if state == "ACTIVE" and baseline_center:
                        if locked_hand_label and current_hand_label != locked_hand_label:
                            continue
                            
                        # Draw Control Box
                        cx, cy = baseline_center
                        top_left = (cx - baseline_box_half_size, cy - baseline_box_half_size)
                        bottom_right = (cx + baseline_box_half_size, cy + baseline_box_half_size)
                        
                        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                        cv2.circle(frame, baseline_center, 5, (255, 0, 0), cv2.FILLED)
                        cv2.putText(frame, "CONTROL BOX", (top_left[0], top_left[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                        # Draw fingertips
                        for i, index in enumerate(fingertip_indices):
                            landmark = hand_landmarks[index]
                            lx, ly = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (lx, ly), 10, (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, finger_names[i], (lx - 15, ly - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        # GRIPPER CONTROL
                        thumb_tip = hand_landmarks[4]
                        index_tip = hand_landmarks[8]
                        
                        thumb_x, thumb_y = thumb_tip.x * w, thumb_tip.y * h
                        index_x, index_y = index_tip.x * w, index_tip.y * h
                        
                        pinch_distance = math.hypot(thumb_x - index_x, thumb_y - index_y)
                        box_width = baseline_box_half_size * 2
                        
                        # --- CONTINUOUS GRIPPER CONTROL LOGIC with SMOOTHING ---
                        # Map pinch distance from [0.25*width to 0.75*width] into [100.0 to 0.0]
                        normalized_pinch = (pinch_distance - 0.25 * box_width) / (0.5 * box_width)
                        normalized_pinch = max(0.0, min(1.0, normalized_pinch))
                        
                        # Target is inverted: 100 (closed) when pinch is small, 0 (open) when pinch is large
                        target_gripper = 100.0 - (normalized_pinch * 100.0)
                        
                        # Apply with exponential smoothing (alpha 0.15 makes it smooth out jitter)
                        robot.set_gripper(target_gripper, alpha=0.15)
                        
                cv2.putText(frame, f"STATE: {state}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if state == "ACTIVE" else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Fingertip Detector', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
