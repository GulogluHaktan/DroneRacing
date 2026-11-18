"""Diagnostic: which part of the vision student is the bottleneck?

Student input = IMU (gyro + accelerometer), previous action, and the *perception outputs a perfect
vision front-end would provide* (noisy gate vectors, obstacles, marker tokens). No velocity, no
attitude, no vehicle parameters. If DAgger works here, the failure of the RGB student is image
understanding; if it fails here too, velocity/attitude are unobservable for this student design.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="runs/t1")
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--beta-iters", type=int, default=60)
    p.add_argument("--give-vel", action="store_true", help="also give body velocity (EKF-like)")
    p.add_argument("--give-att", action="store_true", help="also give attitude matrix")
    a = p.parse_args()
    import json
    meta = json.loads((Path(a.teacher) / "config.json").read_text())
    cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
    cfg.mid_init_prob = 0.5
    env = RaceEnv(cfg, seed=0)
    dev = env.device
    teacher = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
    teacher.load_state_dict(torch.load(Path(a.teacher) / "model.pt", map_location=dev)["ac"])
    teacher.eval()

    def student_in():
        head, gates, obst, tokens = env._parts(True)  # noisy estimator / perception errors
        vel, rot, omega, prev = head
        imu = env.imu_obs()
        parts = [imu, env.prev_action] + gates + obst + tokens
        if cfg.landing:
            parts.append(env.in_landing().float()[:, None])
        if a.give_vel:
            parts.append(vel)
        if a.give_att:
            parts.append(rot)
        return torch.cat(parts, -1)

    d_in = student_in().shape[-1]
    gru = nn.GRU(d_in, 128, batch_first=True).to(dev)
    pre = nn.Sequential(nn.Linear(d_in, d_in), nn.Tanh()).to(dev)
    head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4)).to(dev)
    params = list(gru.parameters()) + list(pre.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def fwd(x, h=None):
        y, h = gru(pre(x), h)
        return torch.tanh(head(y)), h

    n, T = env.n, a.T
    h = torch.zeros(1, n, 128, device=dev)
    buf = []
    for it in range(1, a.iters + 1):
        beta = max(0.0, 1 - (it - 1) / a.beta_iters)
        xs, ys = [], []
        crash = fin = 0.0
        cnt = 0
        with torch.no_grad():
            for _ in range(T):
                x = student_in()
                t_act = teacher.actor(teacher.pn(env.teacher_obs())).clamp(-1, 1)
                s_act, h = fwd(x[:, None], h)
                use_t = (torch.rand(n, device=dev) < beta)[:, None]
                act = torch.where(use_t, t_act, s_act[:, 0])
                xs.append(x); ys.append(t_act)
                _, _, _, _, done, info = env.step(act)
                h = h * (~done).float()[None, :, None]
                crash += info["crashed"].float().sum().item(); fin += info["finished"].float().sum().item()
                cnt += int(done.sum().item())
        buf.append((torch.stack(xs, 1), torch.stack(ys, 1)))
        buf = buf[-20:]
        X = torch.cat([b[0] for b in buf]); Y = torch.cat([b[1] for b in buf])
        loss_m = 0.0
        for _ in range(24):
            ii = torch.randint(0, X.shape[0], (512,), device=dev)
            pred, _ = fwd(X[ii])
            loss = F.mse_loss(pred[:, 8:], Y[ii][:, 8:])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            loss_m += loss.item() / 24
        if it % 10 == 0 or it == 1:
            c = max(cnt, 1)
            print(f"it {it:4d} | beta {beta:.2f} | bc {loss_m:.4f} | crash {crash/c*100:3.0f}% | fin {fin/c*100:3.0f}%", flush=True)


if __name__ == "__main__":
    main()
