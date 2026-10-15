import dataclasses
from typing import Dict, Optional, Tuple
import numpy as np

@dataclasses.dataclass
class SafetyReport:
    intervened: bool
    max_vel_scale_applied: float
    jerk_clamped: bool
    energy_exceeded: bool
    kinetic_energy_joules: float
    violation_reason: Optional[str] = None
    plausibility_score: float = 1.0

@dataclasses.dataclass
class RobotLimits:
    name: str
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    ddq_max: np.ndarray
    jerk_max: np.ndarray
    link_masses: np.ndarray
    max_collab_energy_j: float

PRESETS: Dict[str, RobotLimits] = {
    "franka_panda": RobotLimits(
        name="franka_panda",
        q_min=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
        q_max=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
        dq_max=np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
        ddq_max=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
        jerk_max=np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]),
        link_masses=np.array([4.97, 0.64, 3.22, 3.58, 1.22, 1.66, 0.73]),
        max_collab_energy_j=4.5,
    ),
    "unitree_g1_arm": RobotLimits(
        name="unitree_g1_arm",
        q_min=np.array([-2.5, -1.5, -2.5, -2.2, -1.8]),
        q_max=np.array([2.5, 1.5, 2.5, 0.1, 1.8]),
        dq_max=np.array([3.5, 3.5, 4.0, 4.0, 5.0]),
        ddq_max=np.array([20.0, 20.0, 25.0, 25.0, 30.0]),
        jerk_max=np.array([8000.0, 8000.0, 9000.0, 9000.0, 12000.0]),
        link_masses=np.array([2.1, 1.8, 1.4, 0.9, 0.5]),
        max_collab_energy_j=3.5,
    )
}

class ActionFirewall:
    def __init__(self, limits: RobotLimits, dt: float = 0.02):
        self.limits = limits
        self.dt = dt

    @classmethod
    def from_preset(cls, preset_name: str, dt: float = 0.02) -> "ActionFirewall":
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset {preset_name}")
        return cls(limits=PRESETS[preset_name], dt=dt)

    def step(
        self,
        action_chunk: np.ndarray,
        current_q: np.ndarray,
        current_dq: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, SafetyReport]:
        action_chunk = np.atleast_2d(action_chunk).copy()
        horizon, dof = action_chunk.shape
        if current_dq is None:
            current_dq = np.zeros(dof)

        safe_chunk = np.zeros_like(action_chunk)
        q_prev = current_q.copy()
        dq_prev = current_dq.copy()
        ddq_prev = np.zeros(dof)

        intervened = False
        jerk_clamped = False
        energy_exceeded = False
        max_vel_scale = 1.0
        peak_energy = 0.0
        reasons = []

        for t in range(horizon):
            q_target = action_chunk[t]
            q_clamped = np.clip(q_target, self.limits.q_min, self.limits.q_max)
            if not np.allclose(q_clamped, q_target, atol=1e-4):
                intervened = True
                if "pos_limit" not in reasons:
                    reasons.append("pos_limit")

            dq_req = (q_clamped - q_prev) / self.dt
            ddq_req = (dq_req - dq_prev) / self.dt
            jerk_req = (ddq_req - ddq_prev) / self.dt

            jerk_safe = np.clip(jerk_req, -self.limits.jerk_max, self.limits.jerk_max)
            if not np.allclose(jerk_safe, jerk_req, atol=1e-2):
                intervened = True
                jerk_clamped = True
                if "jerk_hallucination" not in reasons:
                    reasons.append("jerk_hallucination")

            ddq_safe = np.clip(ddq_prev + jerk_safe * self.dt, -self.limits.ddq_max, self.limits.ddq_max)
            dq_safe = dq_prev + ddq_safe * self.dt

            vel_ratios = np.abs(dq_safe) / self.limits.dq_max
            max_ratio = np.max(vel_ratios)
            if max_ratio > 1.0:
                dq_safe = dq_safe / max_ratio
                max_vel_scale = max(max_vel_scale, float(max_ratio))
                intervened = True
                if "velocity_exceeded" not in reasons:
                    reasons.append("velocity_exceeded")

            est_energy = 0.5 * float(np.sum(self.limits.link_masses * (dq_safe ** 2) * 0.25))
            peak_energy = max(peak_energy, est_energy)
            if est_energy > self.limits.max_collab_energy_j:
                scale = np.sqrt(self.limits.max_collab_energy_j / (est_energy + 1e-8))
                dq_safe *= scale
                intervened = True
                energy_exceeded = True
                if "iso15066_energy_cap" not in reasons:
                    reasons.append("iso15066_energy_cap")

            q_safe = np.clip(q_prev + dq_safe * self.dt, self.limits.q_min, self.limits.q_max)
            safe_chunk[t] = q_safe
            q_prev = q_safe
            dq_prev = dq_safe
            ddq_prev = ddq_safe

        deviation = np.linalg.norm(action_chunk - safe_chunk) / (np.linalg.norm(action_chunk) + 1e-6)
        plausibility = float(np.clip(1.0 - deviation, 0.0, 1.0))

        report = SafetyReport(
            intervened=intervened,
            max_vel_scale_applied=round(max_vel_scale, 3),
            jerk_clamped=jerk_clamped,
            energy_exceeded=energy_exceeded,
            kinetic_energy_joules=round(peak_energy, 3),
            violation_reason=", ".join(reasons) if reasons else None,
            plausibility_score=round(plausibility, 4),
        )
        return safe_chunk, report
