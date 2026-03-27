import cv2
import math
from detector import HandGestureDetector
from robot_controller import RobotController

# --- CONFIGURATION ---
MIRROR_VIDEO = True  # Set to True if your camera is physically mirrored
# ---------------------

def main():
    detector = HandGestureDetector(model_path="hand_landmarker.task")

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
                    
                    # If video is mirrored, MediaPipe's reported handedness is flipped
                    if MIRROR_VIDEO and current_hand_label != "Unknown":
                        current_hand_label = "Left" if current_hand_label == "Right" else "Right"
                    
                    if state == "IDLE":
                        if detector.is_fist(hand_landmarks, w, h):
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

                        # Get and Draw fingertips using the API
                        tips = detector.get_fingertips(hand_landmarks, w, h)
                        for name, (lx, ly) in tips.items():
                            # VISUAL FEEDBACK: Change color to red if outside the control box
                            is_outside = (lx < top_left[0] or lx > bottom_right[0] or 
                                          ly < top_left[1] or ly > bottom_right[1])
                            dot_color = (0, 0, 255) if is_outside else (0, 255, 0)
                            
                            cv2.circle(frame, (lx, ly), 10, dot_color, cv2.FILLED)
                            cv2.putText(frame, name, (lx - 15, ly - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1, cv2.LINE_AA)
                        
                        # --- RATE-BASED GRIPPER CONTROL with DEADZONE ---
                        thumb_x, thumb_y = tips["Thumb"]
                        index_x, index_y = tips["Index"]
                        pinch_distance = math.hypot(thumb_x - index_x, thumb_y - index_y)
                        box_width = baseline_box_half_size * 2
                        
                        # Normalize pinch: 0.0 (touching) to 1.0 (full box width)
                        norm_pinch = pinch_distance / box_width
                        
                        # Initial target is the current position
                        target_pos = robot.current_action["gripper.pos"]
                        
                        if norm_pinch < 0.25:
                            # CLOSE: The tighter the pinch (<25%), the faster it closes
                            # Speed scale: 0 at 0.25, max at 0.0
                            intensity = (0.25 - norm_pinch) / 0.25
                            target_pos += (5.0 * intensity) # Max 5 units per frame closure
                        elif norm_pinch > 0.60:
                            # OPEN: The wider the hand (>60%), the faster it opens
                            # Speed scale: 0 at 0.60, max at 1.0 (clamped)
                            intensity = (min(1.0, norm_pinch) - 0.60) / 0.25
                            target_pos -= (5.0 * intensity) # Max 5 units per frame opening
                        
                        # Apply with smoothing (reusing the smoothed set_gripper method)
                        robot.set_gripper(target_pos, alpha=0.2)
                        
                        # --- MULTI-JOINT CONTROL (Shoulder, Elbow, Wrist) ---
                        # Finger Selection: 1=Shoulder, 2=Elbow, 3=Wrist
                        extended = detector.get_extended_fingers(hand_landmarks, w, h)
                        # We only care about Index, Middle, Ring for joint selection
                        active_fingers = [f for f in extended if f in ["Index", "Middle", "Ring"]]
                        num_active = len(active_fingers)
                        
                        # Trigger bounds are centered at 'cy' and expand as the hand opens
                        base_deadzone = baseline_box_half_size * 0.4
                        dynamic_offset = baseline_box_half_size * 1.2 * norm_pinch
                        total_deadzone = base_deadzone + dynamic_offset
                        
                        upper_bound = cy - total_deadzone
                        lower_bound = cy + total_deadzone - pinch_distance
                        
                        # Visualize trigger bounds (wide lines)
                        line_half_width = max(baseline_box_half_size, int(total_deadzone * 0.5))
                        cv2.line(frame, (cx - line_half_width, int(upper_bound)), 
                                 (cx + line_half_width, int(upper_bound)), (255, 255, 0), 2)
                        cv2.line(frame, (cx - line_half_width, int(lower_bound)), 
                                 (cx + line_half_width, int(lower_bound)), (0, 255, 255), 2)
                        
                        if num_active > 0:
                            # Calculate average Y of active selection fingers
                            avg_y = sum(tips[f][1] for f in active_fingers) / num_active
                            
                            # Joint Mapping
                            if num_active == 1:
                                joint_name = "shoulder_lift.pos"
                                set_func = robot.set_shoulder_lift
                                label = "SHOULDER"
                            elif num_active == 2:
                                joint_name = "elbow_flex.pos"
                                set_func = robot.set_elbow_flex
                                label = "ELBOW"
                            else: # 3 or more
                                joint_name = "wrist_flex.pos"
                                set_func = robot.set_wrist_flex
                                label = "WRIST"
                            
                            cv2.putText(frame, f"CONTROL: {label}", (cx - baseline_box_half_size, int(upper_bound) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            target_pos = robot.current_action[joint_name]
                            if avg_y < upper_bound:
                                # LIFT: Fast if higher
                                intensity = min(1.0, (upper_bound - avg_y) / (baseline_box_half_size * 0.4))
                                target_pos += (3.0 * intensity)
                            elif avg_y > lower_bound:
                                # LOWER: Fast if lower
                                intensity = min(1.0, (avg_y - lower_bound) / (baseline_box_half_size * 0.4))
                                target_pos -= (3.0 * intensity)
                            
                            set_func(target_pos, alpha=0.1)

                        # --- SHOULDER PAN CONTROL ---
                        # Fixed horizontal bounds (not scaling with pinch)
                        deadzone_x = baseline_box_half_size * 0.7
                        left_bound = cx - deadzone_x
                        right_bound = cx + deadzone_x
                        
                        # Determine labels and directions based on MIRROR_VIDEO
                        if MIRROR_VIDEO:
                            left_label, right_label = "RIGHT", "LEFT"
                            pan_left_condition = (min(index_x, thumb_x) < left_bound)
                            pan_right_condition = (max(index_x, thumb_x) > right_bound)
                            left_intensity = (left_bound - min(index_x, thumb_x)) / (baseline_box_half_size * 0.3)
                            right_intensity = (max(index_x, thumb_x) - right_bound) / (baseline_box_half_size * 0.3)
                        else:
                            left_label, right_label = "LEFT", "RIGHT"
                            pan_right_condition = (min(index_x, thumb_x) < left_bound)
                            pan_left_condition = (max(index_x, thumb_x) > right_bound)
                            right_intensity = (left_bound - min(index_x, thumb_x)) / (baseline_box_half_size * 0.3)
                            left_intensity = (max(index_x, thumb_x) - right_bound) / (baseline_box_half_size * 0.3)

                        # Visualize horizontal trigger bounds with correct labels for the user's perspective
                        cv2.line(frame, (int(left_bound), cy - baseline_box_half_size), 
                                 (int(left_bound), cy + baseline_box_half_size), (0, 0, 255), 2)
                        cv2.line(frame, (int(right_bound), cy - baseline_box_half_size), 
                                 (int(right_bound), cy + baseline_box_half_size), (0, 0, 255), 2)
                        cv2.putText(frame, left_label, (int(left_bound) - 40, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        cv2.putText(frame, right_label, (int(right_bound) + 5, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                        pan_target = robot.current_action["shoulder_pan.pos"]
                        if pan_left_condition:
                            # MOVE LEFT (increase pan)
                            intensity = min(1.0, left_intensity)
                            pan_target += (3.0 * intensity)
                        elif pan_right_condition:
                            # MOVE RIGHT (decrease pan)
                            intensity = min(1.0, right_intensity)
                            pan_target -= (3.0 * intensity)
                        
                        robot.set_shoulder_pan(pan_target, alpha=0.1)

                        # --- WRIST ROLL CONTROL ---
                        # Use angle between thumb and index
                        # baseline is vertical (-90 deg in screen coords)
                        current_angle_rad = math.atan2(index_y - thumb_y, index_x - thumb_x)
                        current_angle_deg = math.degrees(current_angle_rad)
                        
                        # Target is "Straight Line" (vertical = -90)
                        # Deviation from neutral vertical
                        roll_deviation = current_angle_deg + 90
                        # Normalize to -180 to 180
                        roll_deviation = (roll_deviation + 180) % 360 - 180
                        
                        # Visualize Rotation info
                        roll_label = f"ROLL: {roll_deviation:+.1f}"
                        cv2.putText(frame, roll_label, (cx - 40, cy + baseline_box_half_size + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 255), 1)
                        
                        roll_target = robot.current_action["wrist_roll.pos"]
                        # Deadzone of 20 degrees
                        if abs(roll_deviation) > 25:
                            # Intensity based on how much you rotate beyond 20 deg
                            intensity = min(1.0, (abs(roll_deviation) - 20) / 40.0)
                            # Sign: CW rotation (positive deviation) increases roll? 
                            # We'll adjust based on mirror. If MIRROR_VIDEO, CW is CW.
                            if MIRROR_VIDEO:
                                direction = 1 if roll_deviation > 0 else -1
                            else:
                                direction = -1 if roll_deviation > 0 else 1
                                
                            roll_target += (5.0 * intensity * direction)
                        
                        robot.set_wrist_roll(roll_target, alpha=0.1)
                        
                cv2.putText(frame, f"STATE: {state}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if state == "ACTIVE" else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Fingertip Detector', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
