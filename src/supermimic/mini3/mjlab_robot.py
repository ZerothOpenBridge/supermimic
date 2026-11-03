"""Mini3 entity and explicit motor law bound to the frozen evaluator control profile."""
from dataclasses import dataclass
import mujoco
from mjlab.actuator import IdealPdActuator,IdealPdActuatorCfg
from mjlab.entity import EntityCfg,EntityArticulationInfoCfg
from mjlab.utils.spec import create_motor_actuator
from .spec import CONTROL,JOINT_NAMES,DEFAULT_POS,MJCF_PATH
from .motor import MotorTorqueEngine

class Mini3Actuator(IdealPdActuator):
    def edit_spec(self,spec,target_names):
        assert tuple(target_names)==JOINT_NAMES,target_names
        for i,name in enumerate(target_names):
            self._mjs_actuators.append(create_motor_actuator(spec,name,effort_limit=CONTROL['effort_limit'][i],transmission_type=self.cfg.transmission_type))
    def initialize(self,mj_model,model,data,device):
        super().initialize(mj_model,model,data,device)
        self.motor=MotorTorqueEngine(device)
    def compute(self,cmd):
        return self.motor(cmd.position_target,cmd.pos,cmd.vel)

@dataclass(kw_only=True)
class Mini3ActuatorCfg(IdealPdActuatorCfg):
    def build(self,entity,target_ids,target_names):
        return Mini3Actuator(self,entity,target_ids,target_names)

def get_spec():
    spec=mujoco.MjSpec.from_file(MJCF_PATH)
    for a in list(spec.actuators):spec.delete(a)
    passive=CONTROL['passive']
    for i,name in enumerate(JOINT_NAMES):
        j=spec.joint(name)
        j.armature=passive['joint_armature'][i]
        j.frictionloss=passive['joint_frictionloss'][i]
        j.damping=[passive['joint_damping'][i],0.,0.]
    for g in spec.geoms:
        if g.name in ['left_ankle_roll_link_collision','right_ankle_roll_link_collision']:
            g.friction[0]=passive['foot_friction'];g.solref=passive['foot_solref'];g.priority=1
    return spec

def get_robot_cfg():
    return EntityCfg(spec_fn=get_spec,init_state=EntityCfg.InitialStateCfg(pos=(0,0,.5),joint_pos=dict(zip(JOINT_NAMES,DEFAULT_POS)),joint_vel={'.*':0.}),
        articulation=EntityArticulationInfoCfg(actuators=(Mini3ActuatorCfg(target_names_expr=JOINT_NAMES,stiffness=0.,damping=0.,effort_limit=55.),),soft_joint_pos_limit_factor=.9))
