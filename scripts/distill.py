"""DAgger distillation of a trained policy into a smaller actor (same observation layout).

The student actor is a small MLP (deployable on the flight controller); the teacher's full-size critic and observation
normaliser are kept so the student can be fine-tuned with PPO afterwards (train_state.py --init ... --hidden ...).
Run: PYTHONPATH=. ~/isaac-env/bin/python -I scripts/distill.py --teacher runs/h200 --out runs/s200 --hidden 64,64,64
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", default="64,64,64")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--n-envs", type=int, default=2048)
    p.add_argument("--hall-prob", type=float, default=None)
    p.add_argument("--drag-prob", type=float, default=None)
    p.add_argument("--cam-hz", type=float, default=None)
    p.add_argument("--beta-iters", type=int, default=400, help="iterations over which the teacher-control share decays 0.5 -> 0")
    p.add_argument("--replay", type=int, default=8, help="rollouts kept in the training buffer")
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    run = Path(a.teacher)
    meta = json.loads((run / "config.json").read_text())
    args = dict(meta["args"])
    args.update({k: v for k, v in (("hall_prob", a.hall_prob), ("drag_prob", a.drag_prob), ("cam_hz", a.cam_hz)) if v is not None})
    cfg = make_cfg(args, a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
    cfg.soft_frac = 0.0
    cfg.omega_limit = 60.0  # an untrained student can spin the drone out: crash + sanitise non-finite states
    env = RaceEnv(cfg, seed=7)
    dev = cfg.device
    pm = meta["ppo"]
    tcfg = PPOCfg(hidden=tuple(pm["hidden"]), critic_hidden=pm.get("critic_hidden"))
    teacher = ActorCritic(env.policy_dim, env.critic_dim, 4, tcfg).to(dev)
    sd = torch.load(run / "model.pt", map_location=dev)["ac"]
    teacher.load_state_dict(sd)
    teacher.eval()
    hidden = tuple(int(x) for x in a.hidden.split(","))
    student = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg(hidden=hidden, critic_hidden=tcfg.hidden)).to(dev)
    # keep the teacher's normalisers, critic and std; only the actor is new
    for k, v in sd.items():
        if not k.startswith("actor."):
            student.state_dict()[k].copy_(v)
    opt = torch.optim.Adam(student.actor.parameters(), lr=a.lr)
    n_par = sum(x.numel() for x in student.actor.parameters())
    print(f"student actor {hidden}: {n_par} parameters (teacher {sum(x.numel() for x in teacher.actor.parameters())})", flush=True)

    obs, _ = env.compute_obs()
    buf_o, buf_a = [], []
    T = 32
    for it in range(1, a.iters + 1):
        beta = 0.5 * max(0.0, 1.0 - it / a.beta_iters)
        use_t = (torch.rand(env.n, device=dev) < beta)[:, None]
        for _ in range(T):
            with torch.no_grad():
                at = teacher.actor(teacher.pn(obs))
                asd = student.actor(student.pn(obs))
                buf_o.append(obs.clone())
                buf_a.append(at)
                act = torch.where(use_t, at, asd)
                act = act + 0.05 * torch.randn_like(act)  # small exploration noise
            obs, _, _, _, done, info = env.step(act)
        buf_o, buf_a = buf_o[-a.replay * T :], buf_a[-a.replay * T :]
        O, A = torch.cat(buf_o), torch.cat(buf_a)
        ok = torch.isfinite(O).all(-1) & torch.isfinite(A).all(-1)
        O, A = O[ok], A[ok]
        perm = torch.randperm(O.shape[0], device=dev)[: 8 * 16384]
        loss_sum = 0.0
        for i in range(8):
            idx = perm[i * 16384 : (i + 1) * 16384]
            if len(idx) == 0:
                continue
            loss = (student.actor(student.pn(O[idx])) - A[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if it % 50 == 0 or it == 1:
            print(f"it {it:4d} | beta {beta:.2f} | mse {loss_sum/max(1, min(8, len(perm)//16384 + 1)):.5f}", flush=True)
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            meta2 = json.loads(json.dumps(meta))
            meta2["ppo"]["hidden"], meta2["ppo"]["critic_hidden"] = list(hidden), list(tcfg.hidden)
            meta2["args"].update({k: v for k, v in args.items() if k in ("hall_prob", "drag_prob", "cam_hz")})
            meta2["args"]["hidden"] = a.hidden
            meta2["args"]["critic_hidden"] = ",".join(str(x) for x in tcfg.hidden)
            (out / "config.json").write_text(json.dumps(meta2, indent=1))
            torch.save({"ac": student.state_dict(), "it": it}, out / "model.pt")


if __name__ == "__main__":
    main()
