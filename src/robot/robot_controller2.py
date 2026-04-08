import math
import cv2

class RobotController:
    def __init__(self, motors_controller):
        self.robot = motors_controller

    def _set_gripper(self, pose):
        # Example placeholder parsing of the 3D pose (distance from thumb to index for gripper)
        thumb_z, index_z = pose["joint_4"]["z"], pose["joint_8"]["z"]
        thumb_x, index_x = pose["joint_4"]["x"], pose["joint_8"]["x"]
        thumb_y, index_y = pose["joint_4"]["y"], pose["joint_8"]["y"]
        pinch_dist_3d = math.hypot(thumb_x - index_x, thumb_y - index_y, thumb_z - index_z)

        print(f"z: {thumb_z} {index_z}")
        # print(f"x: {thumb_x} {index_x}")
        # print(f"y: {thumb_y} {index_y}")
        
        # Map pinch_dist_3d back to a simple percentage for gripper logic just as a sanity check
        target_pos = max(0.0, min(100.0, (pinch_dist_3d - 0.02) / 0.08 * 100))
        # target_pos = 100.0 - target_pos # invert so pinched == closed
        
        if self.robot:
            self.robot.set_gripper(target_pos, alpha=0.2)
            
    def _set_shoulder_pan(self, pose_landmarks, locked_hand_label, mirror_video, w, h, frame):
        # Semi-mirroring rotation of the arm: if wrist is to the right of the elbow, rotate right.
        if pose_landmarks:
            pl = pose_landmarks
            is_physical_right = (locked_hand_label == "Right")
            
            # MediaPipe Pose labeling on a mirrored frame interprets physical right as left.
            if mirror_video:
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
                
                if self.robot:
                    self.robot.set_shoulder_pan(pan_target, alpha=0.1)

                # Draw a visual line indicating the vector being mapped
                cx_e, cy_e = int(elbow.x * w), int(elbow.y * h)
                cx_w, cy_w = int(wrist.x * w), int(wrist.y * h)
                cv2.line(frame, (cx_e, cy_e), (cx_w, cy_w), (0, 255, 255), 3)
            
    def _set_elbow_flex(self, hand_landmarks, baseline_wrist_y, w, h, frame):
        # --- Elbow Flex Control (Dynamic Sensitivity: Baseline = -10, Camera Bottom = +90) ---
        if baseline_wrist_y is not None:
            # Use hand wrist (landmark 0) for more stable tracking
            hand_wrist_y = hand_landmarks[0].y
            # dy is positive when hand is BELOW baseline
            dy = hand_wrist_y - baseline_wrist_y
            
            # Calculate available space between baseline and camera bottom (1.0)
            available_space = max(0.01, 1.0 - baseline_wrist_y)
            
            # Baseline (dy=0) maps to -10. 
            # Moving DOWN (dy > 0) INCREASES angle towards +90 over available_space.
            if dy > 0:
                # (dy / available_space) is percentage of distance to bottom
                val = -90.0 + (dy / available_space * 1.4) * 180
            else:
                val = -90.0 # At or above baseline, min at -10
            
            # Clamp and apply stepping
            capped_val = max(-90.0, min(90.0, val))
            elbow_flex_target = round(capped_val / 3.0) * 3.0
            
            if self.robot:
                self.robot.set_elbow_flex(elbow_flex_target, alpha=0.1)
            
            # HUD: draw baseline height line
            by_px = int(baseline_wrist_y * h)
            cv2.line(frame, (0, by_px), (w, by_px), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "ELBOW BASE", (10, by_px - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
    def _set_shoulder_lift(self, pose_landmarks, hand_landmarks, baseline_shoulder_dist, locked_hand_label, mirror_video, w, h, frame):
        # --- Shoulder Lift Control (Wrist-to-Elbow Distance) ---
        # If distance decreases (hand brought to elbow), shoulder lift increases.
        if pose_landmarks and baseline_shoulder_dist is not None:
            pl = pose_landmarks
            is_right = (locked_hand_label == "Right")
            if mirror_video:
                elbow_idx = 13 if is_right else 14
            else:
                elbow_idx = 14 if is_right else 13

            if elbow_idx < len(pl):
                elbow_lm = pl[elbow_idx]
                wrist_lm = hand_landmarks[0] # Using Hand landmarker wrist
                
                # Y-axis only distance: wrist above elbow (wrist_lm.y < elbow_lm.y)
                # if wrist is below elbow (wrist_lm.y > elbow_lm.y), dist is 0
                current_dist = max(0, elbow_lm.y - wrist_lm.y)
                
                # Clamp current_dist so it doesn't exceed baseline
                current_dist = min(current_dist, baseline_shoulder_dist)
                
                # Mapping: baseline_dist -> -10,  0.05 dist -> 90
                # min_dist_anchor = 0.05 (normalized units)
                min_dist_anchor = 0.05
                
                # diff = distance reduction from baseline towards min_dist_anchor
                diff = baseline_shoulder_dist - current_dist
                max_diff = max(0.01, baseline_shoulder_dist - min_dist_anchor)
                
                # Scale 0..max_diff to -10..90
                lift_val = -10.0 + (max(0, diff) / max_diff) * 100.0
                
                # Stepping: discrete 3.0 degree increments
                lift_target = round(lift_val / 3.0) * 3.0
                
                if self.robot:
                    self.robot.set_shoulder_lift(lift_target, alpha=0.1)
                
                # HUD: draw vertical distance indicator for feedback
                ex, ey = int(elbow_lm.x * w), int(elbow_lm.y * h)
                wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
                
                # Show vertical project line
                cv2.line(frame, (ex, ey), (ex, wy), (255, 0, 255), 2)
                # Connector to wrist
                cv2.line(frame, (ex, wy), (wx, wy), (150, 0, 150), 1, cv2.LINE_AA)
                
                hud = f"LIFT: {lift_target:+.0f} deg (dy:{current_dist:.2f})"
                cv2.putText(frame, hud, (wx + 15, wy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def moveRobot(self, hand_landmarks, hand_world_landmarks, pose_landmarks, 
                  locked_hand_label, baseline_wrist_y, baseline_shoulder_dist, 
                  mirror_video, w, h, frame, detector):
                  
        # Extract full 3D pose for future holistic mapping
        pose = detector.extract_full_pose(hand_world_landmarks)
        
        # Dispatch specific logic internal wrappers
        self._set_gripper(pose)
        self._set_shoulder_pan(pose_landmarks, locked_hand_label, mirror_video, w, h, frame)
        self._set_elbow_flex(hand_landmarks, baseline_wrist_y, w, h, frame)
        self._set_shoulder_lift(pose_landmarks, hand_landmarks, baseline_shoulder_dist, locked_hand_label, mirror_video, w, h, frame)
