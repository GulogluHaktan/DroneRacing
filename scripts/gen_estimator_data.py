"""Estimator training data from the fast torch env: the state policy flies, an analytic camera+GateNet model produces
detections, IMU / previous command are logged with the true state / gate labels (same fields as isaac/collect_flight.py).

Run: ~/isaac-env/bin/python -I scripts/gen_estimator_data.py --policy runs/t3 --out runs_host/sim0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.detector_model import sim_detections  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.quad import quat_to_rot  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--profile", default=None, help="override track profile (e.g. a2rl_long)")
    ap.add_argument("--n-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--act-noise", type=float, default=0.12, help="extra action noise on a third of the envs (recovery states)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.policy)
    meta = json.loads((run / "config.json").read_text())
    args = dict(meta["args"])
    if a.profile:
        args["profile"] = a.profile
    outs = []
    for ro in range(a.rollouts):
        cfg = make_cfg(args, a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
        cfg.sensors.scale = 1.0
        env = RaceEnv(cfg, seed=a.seed * 100 + ro)
        dev = env.device
        ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
        ac.load_state_dict(torch.load(run / "model.pt", map_location=dev)["ac"])
        ac.eval()
        obs, _ = env.compute_obs()
        noisy = (torch.arange(a.n_envs, device=dev) % 3 == 0)[:, None].float()
        buf = {k: [] for k in ("feat", "imu", "act", "vel", "up", "gate", "done", "valid")}
        last = None
        for step in range(a.steps):
            R = quat_to_rot(env.quad.quat)
            Rt = R.transpose(1, 2)
            lab_g = []
            last = env._last_slot()
            for j in range(2):
                idx = torch.minimum(env.gate_idx + j, last)
                c, nrm, _ = env._gate_frame(idx)
                rel = (Rt @ (c - env.quad.pos).unsqueeze(-1)).squeeze(-1) / 5.0
                nb = (Rt @ nrm.unsqueeze(-1)).squeeze(-1)
                lab_g.append(torch.cat((rel, nb, env.vis_seen[env.arange, idx].float()[:, None]), -1))
            vals = (sim_detections(env), env.imu_obs(), env.prev_action.clone(),
                    (Rt @ env.quad.vel.unsqueeze(-1)).squeeze(-1) / 10.0, R[:, 2, :], torch.stack(lab_g, 1))
            for k, v in zip(buf, vals):
                buf[k].append(v.half().cpu())
            buf["valid"].append(torch.ones(a.n_envs, dtype=torch.bool))
            d = ac.dist(obs)
            act = (d.sample() + torch.randn(a.n_envs, 4, device=dev) * a.act_noise * noisy).clamp(-1, 1)
            obs, _, _, _, done, _ = env.step(act)
            buf["done"].append(done.cpu())
        outs.append({k: torch.stack(v) for k, v in buf.items()})
        print(f"rollout {ro} done", flush=True)
    d = {k: torch.cat([o[k] for o in outs], 1) for k in outs[0]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save(d, Path(a.out) / "data.pt")
    print("samples", d["done"].numel(), "episodes", int(d["done"].sum()), flush=True)


if __name__ == "__main__":
    main()
