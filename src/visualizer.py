import math
import logging
import foxglove
from foxglove.schemas import FrameTransforms, FrameTransform, Vector3, Quaternion
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R

class FoxgloveVisualizer:
    def __init__(self, urdf_path):
        foxglove.set_log_level(logging.WARNING)
        self.robot = URDF.load(urdf_path)
        self.server = foxglove.start_server()
        print("Foxglove server started on ws://localhost:8765")
        
    def update_visualization(self, current_action):
        joint_positions = {}
        
        pan = current_action.get("shoulder_pan.pos", 0.0)
        lift = current_action.get("shoulder_lift.pos", 0.0)
        elbow = current_action.get("elbow_flex.pos", 90.0)
        wflex = current_action.get("wrist_flex.pos", 0.0)
        wroll = current_action.get("wrist_roll.pos", 0.0)
        gripper = current_action.get("gripper.pos", 50.0)
        
        # Convert degrees to radians and map to URDF limits
        joint_positions["shoulder_pan"] = math.radians(pan) * -1.0
        joint_positions["shoulder_lift"] = math.radians(lift)
        joint_positions["elbow_flex"] = math.radians(elbow)
        joint_positions["wrist_flex"] = math.radians(wflex)
        joint_positions["wrist_roll"] = math.radians(wroll)
        # Convert gripper percent to radians depending on urdf mapping
        joint_positions["gripper"] = ((gripper - 10) / 100.0) * math.pi
        
        cfg = {}
        for joint in self.robot.robot.joints:
            if joint.name in joint_positions:
                cfg[joint.name] = joint_positions[joint.name]
            else:
                cfg[joint.name] = 0.0

        self.robot.update_cfg(cfg)

        transforms = []
        transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="base",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        )

        for joint in self.robot.robot.joints:
            parent_link = joint.parent
            child_link = joint.child
            T_local = self.robot.get_transform(frame_to=child_link, frame_from=parent_link)
            trans = T_local[:3, 3]
            quat = R.from_matrix(T_local[:3, :3]).as_quat()

            transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                    rotation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
                )
            )
            
        foxglove.log("/tf", FrameTransforms(transforms=transforms))
