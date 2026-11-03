"""Mini3 articulation with zero-gain effort drives; explicit motor law is evaluated each 2 ms."""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from .mini3.spec import CONTROL,JOINT_NAMES,DEFAULT_POS,TRACKED_BODIES

G1_JOINT_NAMES=list(JOINT_NAMES)  # keep upstream import boundary; these are Mini3 joints.
MJCF_PATH='/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/control/mini3.xml'
USD_PATH='/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/control/usd/mini3/mini3.usda'
URDF_PATH=CONTROL['mjcf'].replace('/mjcf/mini3.xml','/urdf/mini3.urdf')
DEFAULT_JOINT_POS=dict(zip(JOINT_NAMES,DEFAULT_POS))
def default_joint_pos_list():return list(DEFAULT_POS)
def action_scale(name):return .25
def _actuators():
    return {name:IdealPDActuatorCfg(joint_names_expr=[name],stiffness=0.,damping=0.,effort_limit=CONTROL['effort_limit'][i],
                effort_limit_sim=CONTROL['effort_limit'][i],armature=CONTROL['passive']['joint_armature'][i]) for i,name in enumerate(JOINT_NAMES)}

G1_CFG=ArticulationCfg(spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH,variants={'Physics':'mujoco'},activate_contact_sensors=False,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.5),joint_pos=DEFAULT_JOINT_POS,joint_vel={'.*':0.}),
    soft_joint_pos_limit_factor=.9,actuators=_actuators())
