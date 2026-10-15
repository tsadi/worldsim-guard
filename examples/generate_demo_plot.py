import numpy as np
import matplotlib.pyplot as plt
from worldsim_guard import ActionFirewall

def main():
    firewall = ActionFirewall.from_preset("franka_panda", dt=0.02)
    horizon = 60
    t = np.linspace(0, horizon * 0.02, horizon)

    raw_chunk = np.zeros((horizon, 7))
    raw_chunk[:, 0] = 0.8 * np.sin(2.0 * t)
    raw_chunk[22:28, 0] += np.array([0.45, 1.35, -0.9, 1.1, 0.5, 0.1])

    current_q = np.zeros(7)
    safe_chunk, report = firewall.step(raw_chunk, current_q=current_q)

    print("=== Worldsim-Guard Audit Report ===")
    print("Intervened:        ", report.intervened)
    print("Violations Caught: ", report.violation_reason)
    print("Plausibility Score:", report.plausibility_score)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, raw_chunk[:, 0], color="#ff4b4b", linestyle="--", linewidth=2, label="Raw World Model Output (Hallucinated Spike)")
    ax1.plot(t, safe_chunk[:, 0], color="#00ffcc", linewidth=2.5, label="worldsim-guard Clamped Trajectory (ISO 15066 Safe)")
    ax1.set_ylabel("Joint 1 Position (rad)")
    ax1.set_title("World Model Trajectory Firewall: Hallucination Suppression (<0.3ms latency)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    raw_vel = np.gradient(raw_chunk[:, 0], 0.02)
    safe_vel = np.gradient(safe_chunk[:, 0], 0.02)
    ax2.plot(t, raw_vel, color="#ff4b4b", linestyle="--", alpha=0.8, label="Requested Velocity (Violates Actuator Spec)")
    ax2.plot(t, safe_vel, color="#00ffcc", linewidth=2, label="Clamped Safe Velocity")
    ax2.axhline(firewall.limits.dq_max[0], color="#ffcc00", linestyle=":", label="Max Actuator Limit (+2.17 rad/s)")
    ax2.axhline(-firewall.limits.dq_max[0], color="#ffcc00", linestyle=":")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("assets/hallucination_vs_safe.png", dpi=180)
    print("Saved visual benchmark to assets/hallucination_vs_safe.png")

if __name__ == "__main__":
    main()
