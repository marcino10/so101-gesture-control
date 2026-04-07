import cv2
import math
import time
from detector import HandGestureDetector, ArmPoseDetector
from robot_controller import RobotController

# --- CONFIGURATION ---
MIRROR_VIDEO = True  # Set to True if your camera is physically mirrored
# ---------------------

def main():
    detector = HandGestureDetector(model_path="hand_landmarker.task")
    pose_detector = ArmPoseDetector(model_path="pose_landmarker_full.task")

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

            if MIRROR_VIDEO:
                frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Draw 3x3 Grid (subtle overlay)
            for i in range(1, 3):
                # Vertical
                cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (80, 80, 80), 1)
                # Horizontal
                cv2.line(frame, (0, i * h // 3), (w, i * h // 3), (80, 80, 80), 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # In LIVE_STREAM mode, timestamp must be monotonically increasing
            timestamp_ms = int(time.monotonic() * 1000)
            
            try:
                detector.process_async(rgb_frame, timestamp_ms)
                pose_detector.process_async(rgb_frame, timestamp_ms)
            except Exception as e:
                print(f"Async processing error: {e}")
            
            # Fetch latest cached results
            detection_result = detector.latest_result
            pose_result = pose_detector.latest_result
            
            # --- Visualize Pose Landmarks (Shoulders, Elbows, Wrists) ---
            if pose_result and pose_result.pose_landmarks:
                for pose_landmarks in pose_result.pose_landmarks:
                    # Draw Pose Connections
                    pose_connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                    for (i, j) in pose_connections:
                        if i < len(pose_landmarks) and j < len(pose_landmarks):
                            x1, y1 = int(pose_landmarks[i].x * w), int(pose_landmarks[i].y * h)
                            x2, y2 = int(pose_landmarks[j].x * w), int(pose_landmarks[j].y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)

                    # Draw Joints
                    for lm_idx in [11, 12, 13, 14, 15, 16]:
                        if lm_idx < len(pose_landmarks):
                            lm = pose_landmarks[lm_idx]
                            px, py = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (px, py), 8, (255, 100, 0), cv2.FILLED)
                            label = {11:"L-Should", 12:"R-Should", 13:"L-Elbow", 14:"R-Elbow", 15:"L-Wrist", 16:"R-Wrist"}.get(lm_idx, "")
                            cv2.putText(frame, label, (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            # Handle State Reset if hands disappear
            if not detection_result or not detection_result.hand_landmarks:
                missing_frames += 1
                if missing_frames > 20 and state == "ACTIVE":
                    print("\n>>> HAND LOST: Resetting to IDLE mode <<<")
                    state = "IDLE"
                    baseline_center = None
                    locked_hand_label = None
                    detector.activation_tracker.reset()
                
                cv2.putText(frame, f"STATE: {state}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                missing_frames = 0
                
                for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    handedness_list = detection_result.handedness
                    current_hand_label = handedness_list[hand_idx][0].category_name if (handedness_list and len(handedness_list) > hand_idx) else "Unknown"
                    
                    # If video is mirrored, MediaPipe's reported handedness is flipped
                    if MIRROR_VIDEO and current_hand_label != "Unknown":
                        current_hand_label = "Left" if current_hand_label == "Right" else "Right"
                    
                    hand_world_landmarks = detection_result.hand_world_landmarks[hand_idx] if detection_result.hand_world_landmarks else None
                    if not hand_world_landmarks:
                        continue

                    if state == "IDLE":
                        if detector.check_activation_sequence(hand_world_landmarks):
                            center_lm = hand_landmarks[9] 
                            baseline_center = (int(center_lm.x * w), int(center_lm.y * h))
                            wrist = hand_landmarks[0]
                            hand_size = math.hypot((wrist.x * w) - baseline_center[0], (wrist.y * h) - baseline_center[1])
                            
                            baseline_box_half_size = int((hand_size * 2.05) / 2)
                            if baseline_box_half_size < 50:
                                baseline_box_half_size = 50
                                
                            state = "ACTIVE"
                            locked_hand_label = current_hand_label
                            print(f"\n>>> OPEN-CLOSE-OPEN DETECTED: Locked onto {locked_hand_label} Hand. <<<")
                    
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

                        # Draw Hand connections
                        hand_connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
                            (0, 5), (5, 6), (6, 7), (7, 8),      # index
                            (5, 9), (9, 10), (10, 11), (11, 12), # middle
                            (9, 13), (13, 14), (14, 15), (15, 16), # ring
                            (13, 17), (17, 18), (18, 19), (19, 20), # pinky
                            (0, 17) # wrist to pinky base
                        ]
                        for (i, j) in hand_connections:
                            x1, y1 = int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h)
                            x2, y2 = int(hand_landmarks[j].x * w), int(hand_landmarks[j].y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw all landmarks
                        for lm in hand_landmarks:
                            lx, ly = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (lx, ly), 5, (0, 255, 0), cv2.FILLED)
                        
                        # Extract full 3D pose for future holistic mapping
                        pose = detector.extract_full_pose(hand_world_landmarks)
                        
                        # Target joint dictionary to be calculated in the future
                        target_joints = {}
                        
                        # Example placeholder parsing of the 3D pose (distance from thumb to index for gripper)
                        thumb_z, index_z = pose["joint_4"]["z"], pose["joint_8"]["z"]
                        thumb_x, index_x = pose["joint_4"]["x"], pose["joint_8"]["x"]
                        thumb_y, index_y = pose["joint_4"]["y"], pose["joint_8"]["y"]
                        pinch_dist_3d = math.hypot(thumb_x - index_x, thumb_y - index_y, thumb_z - index_z)
                        
                        # Map pinch_dist_3d back to a simple percentage for gripper logic just as a sanity check
                        gripper_target = max(0.0, min(100.0, (pinch_dist_3d - 0.02) / 0.08 * 100))
                        target_joints["gripper.pos"] = 100.0 - gripper_target # invert so pinched == closed
                        
                        # Dispatch all mapped joints back to controller
                        alpha_dict = {"gripper.pos": 0.2}
                        
                        # --- Shoulder Pan Control ---
                        # Semi-mirroring rotation of the arm: if wrist is to the right of the elbow, rotate right.
                        if pose_result and pose_result.pose_landmarks:
                            pl = pose_result.pose_landmarks[0]
                            is_physical_right = (locked_hand_label == "Right")
                            
                            # MediaPipe Pose labeling on a mirrored frame interprets physical right as left.
                            if MIRROR_VIDEO:
                                elbow_idx = 13 if is_physical_right else 14
                                wrist_idx = 15 if is_physical_right else 16
                            else:
                                elbow_idx = 14 if is_physical_right else 13
                                wrist_idx = 16 if is_physical_right else 15
                                
                            if elbow_idx < len(pl) and wrist_idx < len(pl):
                                elbow = pl[elbow_idx]
                                wrist = pl[wrist_idx]
                                
                                # dx tells us horizontal offsets. (Positive means wrist to the right in the image overlay)
                                dx = wrist.x - elbow.x
                                
                                # Typical dx for wrist-elbow rotation spans around roughly -0.15 to +0.15 in image space.
                                # Let's map dx to -90 to 90 degrees.
                                # An external multiplier controls sensitivity.
                                pan_target = (dx / 0.15) * 90.0
                                
                                # If needed, you can invert the pan_target by multiplying by -1. 
                                # Clamping within healthy robot limits.
                                pan_target = max(-90.0, min(90.0, pan_target))
                                
                                target_joints["shoulder_pan.pos"] = pan_target
                                alpha_dict["shoulder_pan.pos"] = 0.1

                                # Draw a visual line indicating the vector being mapped
                                cx_e, cy_e = int(elbow.x * w), int(elbow.y * h)
                                cx_w, cy_w = int(wrist.x * w), int(wrist.y * h)
                                cv2.line(frame, (cx_e, cy_e), (cx_w, cy_w), (0, 255, 255), 3)
                        
                        robot.set_target_joints(target_joints, alpha_dict=alpha_dict)
                        
                cv2.putText(frame, f"STATE: {state}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if state == "ACTIVE" else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Fingertip Detector', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
