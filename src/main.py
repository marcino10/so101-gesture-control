import cv2
import math
import time
from detector import HandGestureDetector, ArmPoseDetector
from robot.motors_controller import MotorsController
from robot.robot_controller import RobotController

# --- CONFIGURATION ---
MIRROR_VIDEO = True  # Set to True if your camera is physically mirrored
CONTROL_SYSTEM = 1
# ---------------------

def main():
    detector = HandGestureDetector(model_path="hand_landmarker.task")
    pose_detector = ArmPoseDetector(model_path="pose_landmarker_full.task")


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam... Press 'q' to quit.")

    start_time = time.time()
    state = "IDLE"
    baseline_center = None
    baseline_box_half_size = 0
    missing_frames = 0
    locked_hand_label = None
    baseline_elbow_dist = None

    # Connect to the motors automatically, using a context manager
    with MotorsController(port="/dev/ttyACM0") as motors:
        robot = RobotController(motors_controller=motors)
        
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
                    joint_ids = [0, 11, 12, 13, 14, 15, 16]
                    for lm_idx in joint_ids:
                        if lm_idx < len(pose_landmarks):
                            lm = pose_landmarks[lm_idx]
                            px, py = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (px, py), 8, (255, 100, 0), cv2.FILLED)

                            cv2.putText(frame, str(lm_idx), (px + 15, py),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            # Handle State Reset if hands disappear
            if not detection_result or not detection_result.hand_landmarks:
                missing_frames += 1
                if missing_frames > 20 and state == "ACTIVE":
                    print("\n>>> HAND LOST: Resetting to IDLE mode <<<")
                    state = "IDLE"
                    baseline_center = None
                    locked_hand_label = None
                    robot.reset_states()
                    baseline_elbow_dist = None
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
                            
                            # Capture baseline elbow flex distance (hand-wrist to pose-elbow)
                            if pose_result and pose_result.pose_landmarks:
                                pl = pose_result.pose_landmarks[0]
                                is_right = (locked_hand_label == "Right")
                                if MIRROR_VIDEO:
                                    elbow_idx = 13 if is_right else 14
                                else:
                                    elbow_idx = 14 if is_right else 13
                                    
                                if elbow_idx < len(pl):
                                    el = pl[elbow_idx]
                                    wr = hand_landmarks[0]
                                    # Y-axis only distance: wrist above elbow (wr.y < el.y)
                                    baseline_elbow_dist = max(0, el.y - wr.y)

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
                        
                        pose_landmarks = pose_result.pose_landmarks[0] if pose_result and pose_result.pose_landmarks else None
                        
                        pose_world_landmarks = pose_result.pose_world_landmarks[0] if pose_result and pose_result.pose_world_landmarks else None
                        
                        robot.moveRobot(
                            system=CONTROL_SYSTEM,
                            hand_landmarks=hand_landmarks,
                            hand_world_landmarks=hand_world_landmarks,
                            pose_landmarks=pose_landmarks,
                            pose_world_landmarks=pose_world_landmarks,
                            locked_hand_label=locked_hand_label,
                            baseline_elbow_dist=baseline_elbow_dist,
                            mirror_video=MIRROR_VIDEO,
                            w=w,
                            h=h,
                            frame=frame,
                            detector=detector,
                            baseline_center=baseline_center,
                            baseline_box_half_size=baseline_box_half_size
                        )
                        
                cv2.putText(frame, f"STATE: {state}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if state == "ACTIVE" else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Fingertip Detector', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
