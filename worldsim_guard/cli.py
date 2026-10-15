import argparse
import time
import numpy as np
from .firewall import ActionFirewall, PRESETS

def run_benchmark(preset: str, rollouts: int = 100, horizon: int = 50):
    firewall = ActionFirewall.from_preset(preset, dt=0.02)
    dof = len(firewall.limits.q_min)
    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    interventions = 0
    jerk_violations = 0
    energy_violations = 0
    scores = []

    for _ in range(rollouts):
        t = np.linspace(0, horizon * 0.02, horizon)
        chunk = np.zeros((horizon, dof))
        for j in range(dof):
            chunk[:, j] = 0.5 * np.sin(2.0 * t + rng.uniform(0, 3.14))
        if rng.random() < 0.35:
            spike_idx = rng.integers(5, horizon - 5)
            chunk[spike_idx:spike_idx+3, 0] += rng.uniform(-1.5, 1.5, size=3)

        _, report = firewall.step(chunk, current_q=chunk[0])
        if report.intervened:
            interventions += 1
        if report.jerk_clamped:
            jerk_violations += 1
        if report.energy_exceeded:
            energy_violations += 1
        scores.append(report.plausibility_score)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_step_us = (elapsed_ms * 1000.0) / (rollouts * horizon)

    print(f"\n🛡️  worldsim-guard Physical Plausibility Benchmark ({preset})")
    print("=" * 64)
    print(f"Rollouts Evaluated:        {rollouts} trajectories ({rollouts * horizon} steps)")
    print(f"Mean Filtering Latency:    {elapsed_ms / rollouts:.3f} ms/chunk ({per_step_us:.1f} µs/step)")
    print(f"Hallucinations Caught:     {interventions}/{rollouts} ({interventions/rollouts*100:.1f}%)")
    print(f"Jerk Limit Violations:     {jerk_violations}/{rollouts} ({jerk_violations/rollouts*100:.1f}%)")
    print(f"ISO 15066 Energy Caps:     {energy_violations}/{rollouts} ({energy_violations/rollouts*100:.1f}%)")
    print(f"Mean Plausibility Score:   {np.mean(scores):.4f} / 1.0000")
    print("=" * 64)

def main():
    parser = argparse.ArgumentParser(description="worldsim-guard CLI audit & benchmark tool")
    parser.add_argument("command", choices=["benchmark", "presets"], help="Command to run")
    parser.add_argument("--preset", default="franka_panda", help="Hardware preset name")
    parser.add_argument("--rollouts", type=int, default=200, help="Number of rollouts")
    args = parser.parse_args()

    if args.command == "presets":
        print("Available hardware safety presets:", ", ".join(PRESETS.keys()))
    elif args.command == "benchmark":
        run_benchmark(args.preset, args.rollouts)

if __name__ == "__main__":
    main()
