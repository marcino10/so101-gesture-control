from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower import SOFollowerRobotConfig

class RobotController:
    def __init__(self, port="/dev/ttyACM0"):
        self.config = SOFollowerRobotConfig(
            port=port,
            id="so101_main"
        )
        self.robot = SOFollower(self.config)
        self.current_action = {}
        
    def __enter__(self):
        try:
            self.robot.connect()
            print("Connected to SO-101 robot!")
            obs = self.robot.get_observation()
            
            # Initialize default action to current observation to avoid sudden jerks.
            # Replace missing observations with sensible defaults.
            self.current_action = {
                "shoulder_pan.pos": obs.get("shoulder_pan.pos", 0.0) if obs else 0.0,
                "shoulder_lift.pos": obs.get("shoulder_lift.pos", -45.0) if obs else -45.0,
                "elbow_flex.pos": obs.get("elbow_flex.pos", 90.0) if obs else 90.0,
                "wrist_flex.pos": obs.get("wrist_flex.pos", 0.0) if obs else 0.0,
                "wrist_roll.pos": obs.get("wrist_roll.pos", 0.0) if obs else 0.0,
                "gripper.pos": obs.get("gripper.pos", 50.0) if obs else 50.0
            }
        except Exception as e:
            print(f"Warning: Failed to connect to robot on {self.config.port}. Running in simulation mode. Error: {e}")
            self.robot = None
            self.current_action = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": -45.0,
                "elbow_flex.pos": 90.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 50.0
            }
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.robot:
            self.robot.disconnect()
            print("Disconnected from SO-101 robot.")
        
    def set_gripper(self, target_pos, alpha=0.2):
        # Apply exponential moving average smoothing
        current = self.current_action["gripper.pos"]
        smoothed_pos = (alpha * target_pos) + ((1.0 - alpha) * current)
        
        # Clamp between 0 (open) and 100 (closed)
        smoothed_pos = max(0.0, min(100.0, smoothed_pos))
        
        # Only send command if position meaningfully changed to avoid spamming the bus
        if abs(smoothed_pos - current) > 0.5:
            self.current_action["gripper.pos"] = smoothed_pos
            if self.robot:
                self.robot.send_action(self.current_action)
            print(f"Gripper Pos Updated: {smoothed_pos:.1f} (Target: {target_pos:.1f})", flush=True)

    def set_shoulder_lift(self, target_pos, alpha=0.1):
        # Apply exponential moving average smoothing
        current = self.current_action["shoulder_lift.pos"]
        smoothed_pos = (alpha * target_pos) + ((1.0 - alpha) * current)
        
        # Clamp between -90 and 90 (typical range for SO-101 shoulder lift)
        smoothed_pos = max(-90.0, min(90.0, smoothed_pos))
        
        # Only send command if position meaningfully changed
        if abs(smoothed_pos - current) > 0.1:
            self.current_action["shoulder_lift.pos"] = smoothed_pos
            if self.robot:
                self.robot.send_action(self.current_action)
            print(f"Shoulder Lift Updated: {smoothed_pos:.1f} (Target: {target_pos:.1f})", flush=True)

    def set_elbow_flex(self, target_pos, alpha=0.1):
        # Apply exponential moving average smoothing
        current = self.current_action["elbow_flex.pos"]
        smoothed_pos = (alpha * target_pos) + ((1.0 - alpha) * current)
        
        # Clamp between 0 and 180 (typical range for SO-101 elbow)
        smoothed_pos = max(0.0, min(180.0, smoothed_pos))
        
        # Only send command if position meaningfully changed
        if abs(smoothed_pos - current) > 0.1:
            self.current_action["elbow_flex.pos"] = smoothed_pos
            if self.robot:
                self.robot.send_action(self.current_action)
            print(f"Elbow Flex Updated: {smoothed_pos:.1f} (Target: {target_pos:.1f})", flush=True)

    def set_wrist_flex(self, target_pos, alpha=0.1):
        # Apply exponential moving average smoothing
        current = self.current_action["wrist_flex.pos"]
        smoothed_pos = (alpha * target_pos) + ((1.0 - alpha) * current)
        
        # Clamp between -90 and 90 (typical range for SO-101 wrist flex)
        smoothed_pos = max(-90.0, min(90.0, smoothed_pos))
        
        # Only send command if position meaningfully changed
        if abs(smoothed_pos - current) > 0.1:
            self.current_action["wrist_flex.pos"] = smoothed_pos
            if self.robot:
                self.robot.send_action(self.current_action)
            print(f"Wrist Flex Updated: {smoothed_pos:.1f} (Target: {target_pos:.1f})", flush=True)

    def set_shoulder_pan(self, target_pos, alpha=0.1):
        # Apply exponential moving average smoothing
        current = self.current_action["shoulder_pan.pos"]
        smoothed_pos = (alpha * target_pos) + ((1.0 - alpha) * current)
        
        # Clamp between -90 and 90 (typical range for SO-101 shoulder pan)
        smoothed_pos = max(-90.0, min(90.0, smoothed_pos))
        
        # Only send command if position meaningfully changed
        if abs(smoothed_pos - current) > 0.1:
            self.current_action["shoulder_pan.pos"] = smoothed_pos
            if self.robot:
                self.robot.send_action(self.current_action)
            print(f"Shoulder Pan Updated: {smoothed_pos:.1f} (Target: {target_pos:.1f})", flush=True)

    def set_wrist_roll(self, target_pos, alpha=0.1):
        # Apply exponential moving average smoothing
        current = self.current_action["wrist_roll.pos"]
        smoothed_pos = (alpha * target_pos) + ((1.0 - alpha) * current)
        
        # Clamp between -180 and 180 (typical range for SO-101 wrist roll)
        smoothed_pos = max(-180.0, min(180.0, smoothed_pos))
        
        # Only send command if position meaningfully changed
        if abs(smoothed_pos - current) > 0.1:
            self.current_action["wrist_roll.pos"] = smoothed_pos
            if self.robot:
                self.robot.send_action(self.current_action)
            print(f"Wrist Roll Updated: {smoothed_pos:.1f} (Target: {target_pos:.1f})", flush=True)
