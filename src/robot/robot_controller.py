import math
import cv2

class RobotController:
    def __init__(self, motors_controller):
        self.robot = motors_controller
        self.baseline_shoulder_dist_3d = None
        self.prev_rel_depth_3d = None

    def reset_states(self):
        self.baseline_shoulder_dist_3d = None
        self.prev_rel_depth_3d = None

    def _set_gripper(self, pose):
        # Example placeholder parsing of the 3D pose (distance from thumb to index for gripper)
        thumb_z, index_z = pose["joint_4"]["z"], pose["joint_8"]["z"]
        thumb_x, index_x = pose["joint_4"]["x"], pose["joint_8"]["x"]
        thumb_y, index_y = pose["joint_4"]["y"], pose["joint_8"]["y"]
        pinch_dist_3d = math.hypot(thumb_x - index_x, thumb_y - index_y, thumb_z - index_z)

        # print(f"z: {thumb_z} {index_z}")
        
        # Map pinch_dist_3d back to a simple percentage for gripper logic just as a sanity check
        target_pos = max(0.0, min(100.0, (pinch_dist_3d - 0.02) / 0.08 * 100))
        
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
                pan_target = (dx / 0.15) * 90.0
                pan_target = max(-90.0, min(90.0, pan_target))
                
                if self.robot:
                    self.robot.set_shoulder_pan(pan_target, alpha=0.1)

                # Draw a visual line indicating the vector being mapped
                cx_e, cy_e = int(elbow.x * w), int(elbow.y * h)
                cx_w, cy_w = int(wrist.x * w), int(wrist.y * h)
                cv2.line(frame, (cx_e, cy_e), (cx_w, cy_w), (0, 255, 255), 3)

    def _set_elbow_flex(self, pose_landmarks, hand_landmarks, baseline_elbow_dist, locked_hand_label, mirror_video, w, h, frame):
        # --- Elbow Flex Control (Wrist-to-Elbow Distance Y axis) ---
        if pose_landmarks and baseline_elbow_dist is not None:
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
                current_dist = max(0, elbow_lm.y - wrist_lm.y)
                current_dist = min(current_dist, baseline_elbow_dist)
                
                min_dist_anchor = 0.05
                diff = baseline_elbow_dist - current_dist
                max_diff = max(0.01, baseline_elbow_dist - min_dist_anchor)
                
                # Scale 0..max_diff to -10..90
                flex_val = -10.0 + (max(0, diff) / max_diff) * 100.0
                flex_target = round(flex_val / 3.0) * 3.0
                
                if self.robot:
                    self.robot.set_elbow_flex(flex_target, alpha=0.1)
                
                # HUD
                ex, ey = int(elbow_lm.x * w), int(elbow_lm.y * h)
                wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
                cv2.line(frame, (ex, ey), (ex, wy), (255, 0, 255), 2)
                cv2.line(frame, (ex, wy), (wx, wy), (150, 0, 150), 1, cv2.LINE_AA)
                
                hud = f"ELBOW: {flex_target:+.0f} deg (dy:{current_dist:.2f})"
                cv2.putText(frame, hud, (wx + 15, wy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def _set_shoulder_lift(self, pose_world_landmarks, pose_landmarks, locked_hand_label, mirror_video, w, h, frame):
        # --- Shoulder Lift Control (3D distance: wrist vs shoulder) ---
        if pose_world_landmarks:
            pwl = pose_world_landmarks
            is_physical_right = (locked_hand_label == "Right")
            
            if mirror_video:
                shoulder_idx = 11 if is_physical_right else 12
                wrist_idx = 15 if is_physical_right else 16
            else:
                shoulder_idx = 12 if is_physical_right else 11
                wrist_idx = 16 if is_physical_right else 15

            if shoulder_idx < len(pwl) and wrist_idx < len(pwl):
                shoulder = pwl[shoulder_idx]
                wrist = pwl[wrist_idx]
                
                # Use all 3 coords for 3D distance in meters
                dist_3d = math.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y, wrist.z - shoulder.z)

                if self.baseline_shoulder_dist_3d is None:
                    self.baseline_shoulder_dist_3d = dist_3d

                # Delta Deadzone: only move if change > 0.02 meters
                if self.prev_rel_depth_3d is None or abs(dist_3d - self.prev_rel_depth_3d) > 0.02:
                    self.prev_rel_depth_3d = dist_3d
                        
                # Normalized reach: mapping delta from baseline to [-1.0..1.0]
                delta = self.prev_rel_depth_3d - self.baseline_shoulder_dist_3d
                norm_reach = delta / 0.1
                norm_reach = max(-1.0, min(1.0, norm_reach))

                # Linear mapping for predictability
                # Output range [-10, 90] -> Center: 40, Scale: 50
                val = 40.0 + norm_reach * 50.0

                # Stepping: discrete 3.0 degree increments to reduce jitter
                lift_target = round(val / 3.0) * 3.0
                
                if self.robot:
                    self.robot.set_shoulder_lift(lift_target, alpha=0.05)
                
                # Draw HUD near wrist using image coordinates
                if pose_landmarks and wrist_idx < len(pose_landmarks):
                    pm = pose_landmarks[wrist_idx]
                    wr_px, wr_py = int(pm.x * w), int(pm.y * h)
                    hud = f"lift:{lift_target:+.0f} d3d:{dist_3d:.2f}m"
                    cv2.putText(frame, hud, (wr_px + 15, wr_py + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

    def moveRobot(self, hand_landmarks, hand_world_landmarks, pose_landmarks, pose_world_landmarks,
                  locked_hand_label, baseline_elbow_dist, 
                  mirror_video, w, h, frame, detector):
                  
        # Extract full 3D pose for future holistic mapping
        pose = detector.extract_full_pose(hand_world_landmarks)
        
        # Dispatch specific logic internal wrappers
        self._set_gripper(pose)
        self._set_shoulder_pan(pose_landmarks, locked_hand_label, mirror_video, w, h, frame)
        self._set_elbow_flex(pose_landmarks, hand_landmarks, baseline_elbow_dist, locked_hand_label, mirror_video, w, h, frame)
        self._set_shoulder_lift(pose_world_landmarks, pose_landmarks, locked_hand_label, mirror_video, w, h, frame)
