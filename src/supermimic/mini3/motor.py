from __future__ import annotations
"""Shared Mini3 torque law, matched to the frozen evaluator (response/KT disabled).
Parallel-ankle math derived from the historical Mini3 training actuator; provenance retained.
"""
import math
from types import SimpleNamespace
import torch
from .spec import CONTROL,JOINT_NAMES
from .reference_numpy import ANKLE_PARAMS,MOTOR_SPECS

def tn_clip(torque,velocity,kind):
 spec=MOTOR_SPECS[kind];peak=spec['peak_torque'];speed=velocity*60/(2*math.pi)
 drop=(spec['no_load_speed_rpm']-spec.get('peak_speed_rpm',spec['rated_speed_rpm']))/peak
 lower=torch.maximum(torch.full_like(speed,-peak),(-spec['no_load_speed_rpm']-speed)/drop)
 upper=torch.minimum(torch.full_like(speed,peak),(spec['no_load_speed_rpm']-speed)/drop)
 fallback=torch.where(speed>=0,lower,upper);empty=lower>upper
 return torch.clamp(torque,min=torch.where(empty,fallback,lower),max=torch.where(empty,fallback,upper))

class ParallelAnkle:
    def __init__(self,side,device,dtype):
        self.cfg=SimpleNamespace(side=side,jacobian_epsilon=1e-6,default_motor_kd=.1,max_motor_kd=3.5)
        self.pitch_index=0;self.roll_index=1;self.parameters=dict(ANKLE_PARAMS)
        self.eye2=torch.eye(2,device=device,dtype=dtype)
        ids=[JOINT_NAMES.index(side+'_ankle_'+x+'_joint') for x in ['pitch','roll']]
        self.stiffness=torch.tensor([CONTROL['kp'][i] for i in ids],device=device,dtype=dtype)[None]
        self.damping=torch.tensor([CONTROL['kd'][i] for i in ids],device=device,dtype=dtype)[None]
    def torque(self,target,pos,vel):
        self.stiffness=self.stiffness.expand_as(pos);self.damping=self.damping.expand_as(pos)
        cmd=SimpleNamespace(pos=pos,vel=vel,position_target=target,velocity_target=torch.zeros_like(pos),effort_target=torch.zeros_like(pos))
        side=self._side_motor_command(cmd)
        if self.cfg.side=='left':
            motor=torch.stack((side['tmr'],side['tml']),1);velocity=torch.stack((side['tmr_velocity'],side['tml_velocity']),1)
        else:
            motor=torch.stack((side['tml'],side['tmr']),1);velocity=torch.stack((side['tml_velocity'],side['tmr_velocity']),1)
        return self._motor_to_joint(tn_clip(motor,velocity,'25d'),side)
    def _side_motor_command(self, cmd: ActuatorCmd) -> dict[str, Any]:
        assert self.stiffness is not None
        assert self.damping is not None
        assert self.eye2 is not None
        roll = cmd.pos[:, self.roll_index]
        pitch = cmd.pos[:, self.pitch_index]
        jacobian, valid = self._ankle_jacobian(
            self._side_args(), roll, pitch
        )
        safe_jacobian = torch.where(
            valid[:, None, None], jacobian, self.eye2.expand_as(jacobian)
        )
        inverse = torch.linalg.pinv(safe_jacobian)
        joint_velocity = torch.stack(
            (cmd.vel[:, self.roll_index], cmd.vel[:, self.pitch_index]), dim=1
        )
        motor_velocity = torch.bmm(
            safe_jacobian, joint_velocity.unsqueeze(-1)
        ).squeeze(-1)
        joint_damping = torch.zeros(
            cmd.pos.shape[0], 2, 2, device=cmd.pos.device, dtype=cmd.pos.dtype
        )
        joint_damping[:, 0, 0] = self.damping[:, self.roll_index]
        joint_damping[:, 1, 1] = self.damping[:, self.pitch_index]
        motor_damping = inverse.transpose(1, 2).bmm(joint_damping).bmm(inverse)
        diagonal = torch.nan_to_num(
            torch.diagonal(motor_damping, dim1=1, dim2=2),
            nan=self.cfg.default_motor_kd,
        )
        diagonal = torch.clamp(
            diagonal, min=0.0, max=self.cfg.max_motor_kd
        )
        cross = torch.nan_to_num(motor_damping.clone())
        cross[:, 0, 0] = 0.0
        cross[:, 1, 1] = 0.0
        desired_joint_torque = torch.stack(
            (
                self.stiffness[:, self.roll_index]
                * (cmd.position_target[:, self.roll_index] - roll)
                + self.damping[:, self.roll_index]
                * cmd.velocity_target[:, self.roll_index]
                + cmd.effort_target[:, self.roll_index],
                self.stiffness[:, self.pitch_index]
                * (cmd.position_target[:, self.pitch_index] - pitch)
                + self.damping[:, self.pitch_index]
                * cmd.velocity_target[:, self.pitch_index]
                + cmd.effort_target[:, self.pitch_index],
            ),
            dim=1,
        )
        motor_torque = torch.bmm(
            inverse.transpose(1, 2), desired_joint_torque.unsqueeze(-1)
        ).squeeze(-1)
        motor_torque -= torch.bmm(
            cross, motor_velocity.unsqueeze(-1)
        ).squeeze(-1)
        motor_torque -= diagonal * motor_velocity
        motor_torque = torch.where(
            valid[:, None], motor_torque, torch.zeros_like(motor_torque)
        )
        motor_velocity = torch.where(
            valid[:, None], motor_velocity, torch.zeros_like(motor_velocity)
        )
        return {
            "jacobian": safe_jacobian,
            "valid": valid,
            "tml": motor_torque[:, 0],
            "tmr": motor_torque[:, 1],
            "tml_velocity": motor_velocity[:, 0],
            "tmr_velocity": motor_velocity[:, 1],
        }

    def _motor_to_joint(
        self, motor_torque: torch.Tensor, side: dict[str, Any]
    ) -> torch.Tensor:
        output = torch.zeros_like(motor_torque)
        motor_tml_tmr = (
            motor_torque[:, [1, 0]]
            if self.cfg.side == "left"
            else motor_torque
        )
        joint_roll_pitch = torch.bmm(
            side["jacobian"].transpose(1, 2),
            motor_tml_tmr.unsqueeze(-1),
        ).squeeze(-1)
        joint_roll_pitch = torch.where(
            side["valid"][:, None],
            joint_roll_pitch,
            torch.zeros_like(joint_roll_pitch),
        )
        output[:, self.roll_index] = joint_roll_pitch[:, 0]
        output[:, self.pitch_index] = joint_roll_pitch[:, 1]
        return output

    def _side_args(self) -> tuple[float, ...]:
        params = self.parameters
        zl = params["zl"] + params["z0"]
        zr = params["zr"] + params["z0"]
        if self.cfg.side == "left":
            return (
                params["d"], params["df"], zl, zr, params["l"],
                params["lm"], params["hl"], params["hr"], params["z0"],
            )
        return (
            params["d"], params["df"], zr, zl, params["l"],
            params["lm"], params["hr"], params["hl"], params["z0"],
        )

    def _ankle_jacobian(
        self,
        args: tuple[float, ...],
        roll: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = roll.dtype
        roll = roll.to(dtype=torch.float64)
        pitch = pitch.to(dtype=torch.float64)
        h = self.cfg.jacobian_epsilon
        tmr, tml, valid = self._ankle_ik(args, roll, pitch)
        tmr_roll, tml_roll, valid_roll = self._ankle_ik(
            args, roll + h, pitch
        )
        tmr_pitch, tml_pitch, valid_pitch = self._ankle_ik(
            args, roll, pitch + h
        )
        jacobian = torch.stack(
            (
                torch.stack(
                    ((tml_roll - tml) / h, (tml_pitch - tml) / h), dim=1
                ),
                torch.stack(
                    ((tmr_roll - tmr) / h, (tmr_pitch - tmr) / h), dim=1
                ),
            ),
            dim=1,
        )
        valid &= valid_roll & valid_pitch
        valid &= torch.all(torch.isfinite(jacobian), dim=(1, 2))
        return jacobian.to(dtype=original_dtype), valid

    @staticmethod
    def _ankle_ik(
        args: tuple[float, ...],
        roll: torch.Tensor,
        pitch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d, df, zl, zr, link_l, lm, hl, hr, z0 = args
        cx, cy = torch.cos(roll), torch.cos(pitch)
        sx, sy = torch.sin(roll), torch.sin(pitch)
        epsilon = 1.0e-12
        al = -2.0 * d * (df * cx - z0 * sx)
        bl = 2.0 * d * (zl + link_l * sy - df * sx * cy - z0 * cx * cy)
        cl = (
            d**2
            + (lm - link_l * cy - df * sx * sy - z0 * cx * sy) ** 2
            + (df * cx - z0 * sx) ** 2
            + (zl + link_l * sy - df * sx * cy - z0 * cx * cy) ** 2
        )
        dl = hl**2 - cl
        rl = torch.sqrt(torch.clamp(al**2 + bl**2, min=0.0))
        ar = -2.0 * d * (df * cx + z0 * sx)
        br = -2.0 * d * (zr + link_l * sy + df * sx * cy - z0 * cx * cy)
        cr = (
            d**2
            + (lm - link_l * cy + df * sx * sy - z0 * cx * sy) ** 2
            + (df * cx + z0 * sx) ** 2
            + (zr + link_l * sy + df * sx * cy - z0 * cx * cy) ** 2
        )
        dr = hr**2 - cr
        rr = torch.sqrt(torch.clamp(ar**2 + br**2, min=0.0))
        sl = dl / torch.clamp(rl, min=epsilon)
        sr = dr / torch.clamp(rr, min=epsilon)
        valid_left = (
            (rl + epsilon >= torch.abs(dl))
            & (rl > epsilon)
            & torch.isfinite(sl)
        )
        valid_right = (
            (rr + epsilon >= torch.abs(dr))
            & (rr > epsilon)
            & torch.isfinite(sr)
        )
        tml = torch.asin(torch.clamp(sl, -1.0, 1.0)) - torch.atan2(al, bl)
        tmr = (
            math.pi
            - torch.asin(torch.clamp(sr, -1.0, 1.0))
            - torch.atan2(ar, br)
            - 2.0 * math.pi
        )
        tml = torch.where(valid_left, tml, torch.zeros_like(tml))
        tmr = torch.where(valid_right, tmr, torch.zeros_like(tmr))
        valid = valid_left & valid_right & torch.isfinite(tml) & torch.isfinite(tmr)
        return tmr, tml, valid

class MotorTorqueEngine:
 def __init__(self,device,dtype=torch.float32):
  self.kp=torch.tensor(CONTROL['kp'],device=device,dtype=dtype)
  self.kd=torch.tensor(CONTROL['kd'],device=device,dtype=dtype)
  self.groups={k:[i for i,n in enumerate(JOINT_NAMES) if CONTROL['motor_types'][n]==k and 'ankle' not in n] for k in ['55d','25d','10d']}
  self.ankles={s:ParallelAnkle(s,device,dtype) for s in ['left','right']}
 def __call__(self,target,pos,vel):
  torque=self.kp*(target-pos)-self.kd*vel
  out=torch.zeros_like(torque)
  for kind,ids in self.groups.items():out[:,ids]=tn_clip(torque[:,ids],vel[:,ids],kind)
  for side,ankle in self.ankles.items():
   ids=[JOINT_NAMES.index(side+'_ankle_'+x+'_joint') for x in ['pitch','roll']]
   out[:,ids]=ankle.torque(target[:,ids],pos[:,ids],vel[:,ids])
  return torch.nan_to_num(out)
