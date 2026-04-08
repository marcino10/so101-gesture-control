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

    @staticmethod
    def get_hand_pitch(hand_world_landmarks):
        if not hand_world_landmarks or len(hand_world_landmarks) < 11:
            return 0.0
            
        # Palm vector (Wrist 0 -> MCP 9)
        palm_y = hand_world_landmarks[9].y - hand_world_landmarks[0].y
        palm_xz = math.hypot(hand_world_landmarks[9].x - hand_world_landmarks[0].x, hand_world_landmarks[9].z - hand_world_landmarks[0].z)
        palm_pitch = math.degrees(math.atan2(palm_y, palm_xz))
        
        # Finger vector (MCP 9 -> PIP 10)
        finger_y = hand_world_landmarks[10].y - hand_world_landmarks[9].y
        finger_xz = math.hypot(hand_world_landmarks[10].x - hand_world_landmarks[9].x, hand_world_landmarks[10].z - hand_world_landmarks[9].z)
        finger_pitch = math.degrees(math.atan2(finger_y, finger_xz))
        
        # Average pitch
        return (palm_pitch + finger_pitch) / 2.0

    @staticmethod
    def get_hand_roll(hand_world_landmarks, is_right_hand):
        if not hand_world_landmarks or len(hand_world_landmarks) < 18:
            return 0.0
        dx = hand_world_landmarks[17].x - hand_world_landmarks[5].x
        dy = hand_world_landmarks[17].y - hand_world_landmarks[5].y
        
        if not is_right_hand:
            dx = -dx
            
        return math.degrees(math.atan2(dy, dx))

    def _get_active_fingers(self, pose):
        active = []
        for name, tip_idx, mcp_idx in [("Index", 8, 5), ("Middle", 12, 9), ("Ring", 16, 13)]:
            tip_d = math.hypot(pose[f"joint_{tip_idx}"]["x"], pose[f"joint_{tip_idx}"]["y"])
            mcp_d = math.hypot(pose[f"joint_{mcp_idx}"]["x"], pose[f"joint_{mcp_idx}"]["y"])
            if tip_d > mcp_d + 0.02: 
                active.append(name)
        return active

    def _set_gripper(self, system, pose):
        thumb_z, index_z = pose["joint_4"]["z"], pose["joint_8"]["z"]
        thumb_x, index_x = pose["joint_4"]["x"], pose["joint_8"]["x"]
        thumb_y, index_y = pose["joint_4"]["y"], pose["joint_8"]["y"]
        pinch_dist_3d = math.hypot(thumb_x - index_x, thumb_y - index_y, thumb_z - index_z)

        if system in [2, 3]:
            target_pos = max(0.0, min(100.0, (pinch_dist_3d - 0.02) / 0.08 * 100))
            if self.robot:
                self.robot.set_gripper(target_pos, alpha=0.2)
        elif system == 1:
            norm_pinch = (pinch_dist_3d - 0.02) / 0.08
            if self.robot:
                target_pos = self.robot.current_action.get("gripper.pos", 50.0)
                if norm_pinch < 0.25:
                    intensity = (0.25 - norm_pinch) / 0.25
                    target_pos -= (5.0 * intensity)
                elif norm_pinch > 0.75:
                    intensity = (min(1.0, norm_pinch) - 0.75) / 0.25
                    target_pos += (5.0 * intensity)
                self.robot.set_gripper(target_pos, alpha=0.4)
        return pinch_dist_3d

    def _set_shoulder_pan(self, system, pose_landmarks, locked_hand_label, mirror_video, w, h, frame, thumb_px=0, index_px=0, baseline_box_half_size=100, cx=0, cy=0):
        if system in [2, 3]:
            if pose_landmarks:
                pl = pose_landmarks
                is_physical_right = (locked_hand_label == "Right")
                if mirror_video:
                    elbow_idx = 13 if is_physical_right else 14
                    wrist_idx = 15 if is_physical_right else 16
                else:
                    elbow_idx = 14 if is_physical_right else 13
                    wrist_idx = 16 if is_physical_right else 15
                    
                if elbow_idx < len(pl) and wrist_idx < len(pl):
                    elbow = pl[elbow_idx]
                    wrist = pl[wrist_idx]
                    dx = wrist.x - elbow.x
                    pan_target = (dx / 0.15) * 90.0
                    pan_target = max(-90.0, min(90.0, pan_target))
                    if self.robot:
                        self.robot.set_shoulder_pan(pan_target, alpha=0.1)
                    cx_e, cy_e = int(elbow.x * w), int(elbow.y * h)
                    cx_w, cy_w = int(wrist.x * w), int(wrist.y * h)
                    cv2.line(frame, (cx_e, cy_e), (cx_w, cy_w), (0, 255, 255), 3)

        elif system == 1:
            if baseline_box_half_size:
                deadzone_x = baseline_box_half_size * 0.7
                left_bound = cx - deadzone_x
                right_bound = cx + deadzone_x
                
                if mirror_video:
                    left_label, right_label = "RIGHT", "LEFT"
                    pan_left_condition = (min(index_px, thumb_px) < left_bound)
                    pan_right_condition = (max(index_px, thumb_px) > right_bound)
                    left_intensity = (left_bound - min(index_px, thumb_px)) / (baseline_box_half_size * 0.3)
                    right_intensity = (max(index_px, thumb_px) - right_bound) / (baseline_box_half_size * 0.3)
                else:
                    left_label, right_label = "LEFT", "RIGHT"
                    pan_right_condition = (min(index_px, thumb_px) < left_bound)
                    pan_left_condition = (max(index_px, thumb_px) > right_bound)
                    right_intensity = (left_bound - min(index_px, thumb_px)) / (baseline_box_half_size * 0.3)
                    left_intensity = (max(index_px, thumb_px) - right_bound) / (baseline_box_half_size * 0.3)
    
                cv2.line(frame, (int(left_bound), cy - baseline_box_half_size), 
                         (int(left_bound), cy + baseline_box_half_size), (0, 0, 255), 2)
                cv2.line(frame, (int(right_bound), cy - baseline_box_half_size), 
                         (int(right_bound), cy + baseline_box_half_size), (0, 0, 255), 2)
                cv2.putText(frame, left_label, (int(left_bound) - 40, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(frame, right_label, (int(right_bound) + 5, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
                if self.robot:
                    pan_target = self.robot.current_action.get("shoulder_pan.pos", 0.0)
                    if pan_left_condition:
                        intensity = min(1.0, left_intensity)
                        pan_target -= (5.0 * intensity)
                    elif pan_right_condition:
                        intensity = min(1.0, right_intensity)
                        pan_target += (5.0 * intensity)
                    self.robot.set_shoulder_pan(pan_target, alpha=0.3)

    def _set_elbow_flex(self, system, pose_landmarks, hand_landmarks, baseline_elbow_dist, locked_hand_label, mirror_video, w, h, frame, num_active=0, avg_y=0, upper_bound=0, lower_bound=0, baseline_box_half_size=1, cx=0, baseline_wrist_y=None):
        if system == 3:
            if pose_landmarks and baseline_elbow_dist is not None:
                pl = pose_landmarks
                is_right = (locked_hand_label == "Right")
                if mirror_video:
                    elbow_idx = 13 if is_right else 14
                else:
                    elbow_idx = 14 if is_right else 13

                if elbow_idx < len(pl):
                    elbow_lm = pl[elbow_idx]
                    wrist_lm = hand_landmarks[0]
                    current_dist = max(0, elbow_lm.y - wrist_lm.y)
                    current_dist = min(current_dist, baseline_elbow_dist)
                    
                    min_dist_anchor = 0.05
                    diff = baseline_elbow_dist - current_dist
                    max_diff = max(0.01, baseline_elbow_dist - min_dist_anchor)
                    
                    flex_val = -90.0 + (max(0, diff) / max_diff) * 180.0
                    flex_target = round(flex_val / 3.0) * 3.0
                    
                    if self.robot:
                        self.robot.set_elbow_flex(flex_target, alpha=0.1)
                    
                    ex, ey = int(elbow_lm.x * w), int(elbow_lm.y * h)
                    wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
                    cv2.line(frame, (ex, ey), (ex, wy), (255, 0, 255), 2)
                    cv2.line(frame, (ex, wy), (wx, wy), (150, 0, 150), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"ELBOW: {flex_target:+.0f} deg (dy:{current_dist:.2f})", (wx + 15, wy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        elif system == 2:
            if baseline_wrist_y is not None:
                hand_wrist_y = hand_landmarks[0].y
                dy = hand_wrist_y - baseline_wrist_y
                available_space = max(0.01, 1.0 - baseline_wrist_y)
                
                if dy > 0:
                    val = -90.0 + (dy / available_space * 1.4) * 180
                else:
                    val = -90.0
                
                capped_val = max(-90.0, min(90.0, val))
                elbow_flex_target = round(capped_val / 3.0) * 3.0
                
                if self.robot:
                    self.robot.set_elbow_flex(elbow_flex_target, alpha=0.1)
                
                by_px = int(baseline_wrist_y * h)
                cv2.line(frame, (0, by_px), (w, by_px), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "ELBOW BASE", (10, by_px - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        elif system == 1:
            if num_active == 2 and baseline_box_half_size:
                cv2.putText(frame, "CONTROL: ELBOW", (cx - baseline_box_half_size, int(upper_bound) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if self.robot:
                    target_pos = self.robot.current_action.get("elbow_flex.pos", 0.0)
                    if avg_y < upper_bound:
                        intensity = min(1.0, (upper_bound - avg_y) / (baseline_box_half_size * 0.4))
                        target_pos -= (5.0 * intensity)
                    elif avg_y > lower_bound:
                        intensity = min(1.0, (avg_y - lower_bound) / (baseline_box_half_size * 0.4))
                        target_pos += (5.0 * intensity)
                    self.robot.set_elbow_flex(target_pos, alpha=0.3)

    def _set_shoulder_lift(self, system, pose_world_landmarks, pose_landmarks, locked_hand_label, mirror_video, w, h, frame, num_active=0, avg_y=0, upper_bound=0, lower_bound=0, baseline_box_half_size=1, cx=0, hand_landmarks=None, baseline_elbow_dist=None):
        if system == 3:
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
                    dist_3d = math.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y, wrist.z - shoulder.z)

                    if self.baseline_shoulder_dist_3d is None:
                        self.baseline_shoulder_dist_3d = dist_3d

                    if self.prev_rel_depth_3d is None or abs(dist_3d - self.prev_rel_depth_3d) > 0.02:
                        self.prev_rel_depth_3d = dist_3d
                            
                    delta = self.prev_rel_depth_3d - self.baseline_shoulder_dist_3d
                    norm_reach = delta / 0.1
                    norm_reach = max(-1.0, min(1.0, norm_reach))

                    val = 40.0 + norm_reach * 50.0
                    lift_target = round(val / 3.0) * 3.0
                    
                    if self.robot:
                        self.robot.set_shoulder_lift(lift_target, alpha=0.05)
                    
                    if pose_landmarks and wrist_idx < len(pose_landmarks):
                        pm = pose_landmarks[wrist_idx]
                        wr_px, wr_py = int(pm.x * w), int(pm.y * h)
                        cv2.putText(frame, f"lift:{lift_target:+.0f} d3d:{dist_3d:.2f}m", (wr_px + 15, wr_py + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

        elif system == 2:
            if pose_landmarks and baseline_elbow_dist is not None:
                pl = pose_landmarks
                is_right = (locked_hand_label == "Right")
                if mirror_video:
                    elbow_idx = 13 if is_right else 14
                else:
                    elbow_idx = 14 if is_right else 13

                if elbow_idx < len(pl):
                    elbow_lm = pl[elbow_idx]
                    wrist_lm = hand_landmarks[0]
                    current_dist = max(0, elbow_lm.y - wrist_lm.y)
                    current_dist = min(current_dist, baseline_elbow_dist)
                    
                    min_dist_anchor = 0.05
                    diff = baseline_elbow_dist - current_dist
                    max_diff = max(0.01, baseline_elbow_dist - min_dist_anchor)
                    
                    lift_val = -10.0 + (max(0, diff) / max_diff) * 100.0
                    lift_target = round(lift_val / 3.0) * 3.0
                    
                    if self.robot:
                        self.robot.set_shoulder_lift(lift_target, alpha=0.1)
                    
                    ex, ey = int(elbow_lm.x * w), int(elbow_lm.y * h)
                    wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
                    
                    cv2.line(frame, (ex, ey), (ex, wy), (255, 0, 255), 2)
                    cv2.line(frame, (ex, wy), (wx, wy), (150, 0, 150), 1, cv2.LINE_AA)
                    
                    hud = f"LIFT: {lift_target:+.0f} deg (dy:{current_dist:.2f})"
                    cv2.putText(frame, hud, (wx + 15, wy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        elif system == 1:
            if num_active == 1 and baseline_box_half_size:
                cv2.putText(frame, "CONTROL: SHOULDER", (cx - baseline_box_half_size, int(upper_bound) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if self.robot:
                    target_pos = self.robot.current_action.get("shoulder_lift.pos", 0.0)
                    if avg_y < upper_bound:
                        intensity = min(1.0, (upper_bound - avg_y) / (baseline_box_half_size * 0.4))
                        target_pos += (5.0 * intensity)
                    elif avg_y > lower_bound:
                        intensity = min(1.0, (avg_y - lower_bound) / (baseline_box_half_size * 0.4))
                        target_pos -= (5.0 * intensity)
                    self.robot.set_shoulder_lift(target_pos, alpha=0.3)

    def _set_wrist(self, system, num_active, avg_y, upper_bound, lower_bound, baseline_box_half_size, cx, cy, index_px, index_py, thumb_px, thumb_py, mirror_video, frame, hand_world_landmarks=None, baseline_hand_pitch_y=None, baseline_hand_roll=None, locked_hand_label=None, hand_landmarks=None, w=0, h=0):
        if system == 3:
            if hand_world_landmarks and baseline_hand_pitch_y is not None and baseline_hand_roll is not None:
                # Flex
                current_pitch = self.get_hand_pitch(hand_world_landmarks)
                delta_pitch = current_pitch - baseline_hand_pitch_y
                
                # Default position is -20. Sweeping hand DOWN (positive delta_pitch) increases angle towards +90.
                val_flex = -90.0 + (delta_pitch * 2.0)
                val_flex = max(-90.0, min(90.0, val_flex))
                flex_target = round(val_flex / 3.0) * 3.0
                
                # Roll
                is_right = (locked_hand_label == "Right")
                current_roll = self.get_hand_roll(hand_world_landmarks, is_right)
                delta_roll = (current_roll - baseline_hand_roll + 180) % 360 - 180
                
                val_roll = delta_roll * 1.3
                val_roll = max(-90.0, min(90.0, val_roll))
                roll_target = round(val_roll / 3.0) * 3.0
                
                if self.robot:
                    self.robot.set_wrist_flex(flex_target, alpha=0.1)
                    self.robot.set_wrist_roll(roll_target, alpha=0.1)

                if hand_landmarks:
                    mx, my = int(hand_landmarks[10].x * w), int(hand_landmarks[10].y * h)
                    cv2.putText(frame, f"FLEX: {flex_target:+.0f} ROLL: {roll_target:+.0f}", (mx + 15, my), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        elif system == 1 and baseline_box_half_size:
            if num_active >= 3:
                cv2.putText(frame, "CONTROL: WRIST", (cx - baseline_box_half_size, int(upper_bound) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if self.robot:
                    target_pos = self.robot.current_action.get("wrist_flex.pos", 0.0)
                    if avg_y < upper_bound:
                        intensity = min(1.0, (upper_bound - avg_y) / (baseline_box_half_size * 0.4))
                        target_pos -= (5.0 * intensity)
                    elif avg_y > lower_bound:
                        intensity = min(1.0, (avg_y - lower_bound) / (baseline_box_half_size * 0.4))
                        target_pos += (5.0 * intensity)
                    self.robot.set_wrist_flex(target_pos, alpha=0.3)
            
            # Wrist Roll
            current_angle_rad = math.atan2(index_py - thumb_py, index_px - thumb_px)
            current_angle_deg = math.degrees(current_angle_rad)
            roll_deviation = (current_angle_deg + 90 + 180) % 360 - 180
            
            cv2.putText(frame, f"ROLL: {roll_deviation:+.1f}", (cx - 40, cy + baseline_box_half_size + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 255), 1)
            
            if self.robot:
                roll_target = self.robot.current_action.get("wrist_roll.pos", 0.0)
                if abs(roll_deviation) > 25:
                    intensity = min(1.0, (abs(roll_deviation) - 20) / 40.0)
                    if mirror_video:
                        direction = 1 if roll_deviation > 0 else -1
                    else:
                        direction = -1 if roll_deviation > 0 else 1
                    roll_target -= (8.0 * intensity * direction)
                self.robot.set_wrist_roll(roll_target, alpha=0.3)

    def moveRobot(self, system, hand_landmarks, hand_world_landmarks, pose_landmarks, pose_world_landmarks,
                  locked_hand_label, baseline_elbow_dist, 
                  mirror_video, w, h, frame, detector, baseline_center=None, baseline_box_half_size=100, baseline_wrist_y=None, baseline_hand_pitch_y=None, baseline_hand_roll=None):
                  
        pose = detector.extract_full_pose(hand_world_landmarks)
        
        num_active, avg_y, upper_bound, lower_bound = 0, 0, 0, 0
        cx, cy, index_px, thumb_px, index_py, thumb_py = 0, 0, 0, 0, 0, 0
        
        pinch_dist_3d = self._set_gripper(system, pose)
        
        if system == 1 and baseline_center and baseline_box_half_size:
            cx, cy = baseline_center
            
            thumb_lm, index_lm = hand_landmarks[4], hand_landmarks[8]
            middle_lm, ring_lm = hand_landmarks[12], hand_landmarks[16]
            thumb_px, thumb_py = int(thumb_lm.x * w), int(thumb_lm.y * h)
            index_px, index_py = int(index_lm.x * w), int(index_lm.y * h)
            
            active_fingers = self._get_active_fingers(pose)
            num_active = len(active_fingers)
            
            if num_active > 0:
                y_sum = 0
                for f in active_fingers:
                    if f == "Index": y_sum += int(index_lm.y * h)
                    elif f == "Middle": y_sum += int(middle_lm.y * h)
                    elif f == "Ring": y_sum += int(ring_lm.y * h)
                avg_y = y_sum / num_active

            # Dynamically determine pinch distance by matching logic inside _set_gripper velocity parsing
            norm_pinch = (pinch_dist_3d - 0.02) / 0.08
            
            base_deadzone = baseline_box_half_size * 0.4
            dynamic_offset = baseline_box_half_size * 1.2 * norm_pinch
            total_deadzone = base_deadzone + dynamic_offset
            
            upper_bound = cy - total_deadzone
            lower_bound = cy + total_deadzone - (norm_pinch * baseline_box_half_size * 2)
            
            line_half_width = max(baseline_box_half_size, int(total_deadzone * 0.5))
            cv2.line(frame, (cx - line_half_width, int(upper_bound)), 
                     (cx + line_half_width, int(upper_bound)), (255, 255, 0), 2)
            cv2.line(frame, (cx - line_half_width, int(lower_bound)), 
                     (cx + line_half_width, int(lower_bound)), (0, 255, 255), 2)
                     
            # draw tips inside
            top_left = (cx - baseline_box_half_size, cy - baseline_box_half_size)
            bottom_right = (cx + baseline_box_half_size, cy + baseline_box_half_size)
            
            tips_coords = {
                "Thumb": (thumb_px, thumb_py),
                "Index": (index_px, index_py),
                "Middle": (int(middle_lm.x * w), int(middle_lm.y * h)),
                "Ring": (int(ring_lm.x * w), int(ring_lm.y * h)),
                "Pinky": (int(hand_landmarks[20].x * w), int(hand_landmarks[20].y * h))
            }
            
            for name, (lx, ly) in tips_coords.items():
                is_outside = (lx < top_left[0] or lx > bottom_right[0] or 
                              ly < top_left[1] or ly > bottom_right[1])
                dot_color = (0, 0, 255) if is_outside else (0, 255, 0)
                cv2.circle(frame, (lx, ly), 10, dot_color, cv2.FILLED)
                cv2.putText(frame, name, (lx - 15, ly - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1, cv2.LINE_AA)

        self._set_shoulder_pan(system, pose_landmarks, locked_hand_label, mirror_video, w, h, frame, thumb_px, index_px, baseline_box_half_size, cx, cy)
        self._set_elbow_flex(system, pose_landmarks, hand_landmarks, baseline_elbow_dist, locked_hand_label, mirror_video, w, h, frame, num_active, avg_y, upper_bound, lower_bound, baseline_box_half_size, cx, baseline_wrist_y=baseline_wrist_y)
        self._set_shoulder_lift(system, pose_world_landmarks, pose_landmarks, locked_hand_label, mirror_video, w, h, frame, num_active, avg_y, upper_bound, lower_bound, baseline_box_half_size, cx, hand_landmarks=hand_landmarks, baseline_elbow_dist=baseline_elbow_dist)
        self._set_wrist(system, num_active, avg_y, upper_bound, lower_bound, baseline_box_half_size, cx, cy, index_px, index_py, thumb_px, thumb_py, mirror_video, frame, hand_world_landmarks=hand_world_landmarks, baseline_hand_pitch_y=baseline_hand_pitch_y, baseline_hand_roll=baseline_hand_roll, locked_hand_label=locked_hand_label, hand_landmarks=hand_landmarks, w=w, h=h)
