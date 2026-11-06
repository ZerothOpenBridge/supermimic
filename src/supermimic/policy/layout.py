"""Mini3 proprioception contract; kept independent of simulation imports."""

OBS_TERMS = (
    ("joint_pos", 21), ("joint_vel", 21), ("base_ang_vel", 3),
    ("projected_gravity", 3), ("actions", 21),
)
