import json
from pathlib import Path
CONTROL_PATH=Path("/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/control/control.json")
CONTROL=json.loads(CONTROL_PATH.read_text())
JOINT_NAMES=tuple(CONTROL['joint_names'])
DEFAULT_POS=tuple(CONTROL['default_pos'])
MJCF_PATH=CONTROL['mjcf']
TRACKED_BODIES=('base_link','left_hip_roll_link','left_knee_pitch_link','left_ankle_roll_link','right_hip_roll_link','right_knee_pitch_link','right_ankle_roll_link','waist_yaw_link','left_shoulder_roll_link','left_elbow_pitch_link','left_shoulder_yaw_link','right_shoulder_roll_link','right_elbow_pitch_link','right_shoulder_yaw_link')
def load_model():
 import mujoco
 return mujoco.MjModel.from_xml_path(MJCF_PATH)
