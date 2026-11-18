"""Where do opponent collisions happen? (surrogate, camera-line env). Snapshot taken one policy step BEFORE the contact.
usage: python scripts/opp_anatomy.py --run runs/o1 [--n-envs 2048]"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing import RaceEnv
from droneracing.config import make_cfg
from droneracing.ekf_loop import EkfLoopEnv
from droneracing.ppo import ActorCritic, PPOCfg
from droneracing.quad import quat_to_rot

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n-envs", type=int, default=2048); ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--opp-prob", type=float, default=1.0)
a = ap.parse_args()
run = Path(a.run); meta = json.loads((run / "config.json").read_text()); args = dict(meta["args"])
args.update(dict(fixed_prob=1.0, fixed_jitter=[0.3, 0.05], opp_prob=a.opp_prob, ekf_loop=True))
cfg = make_cfg(args, a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"]); cfg.mid_init_prob = 0.0
env = EkfLoopEnv(RaceEnv(cfg, seed=a.seed))
pm = meta.get("ppo", {})
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg(hidden=tuple(pm.get("hidden", (256, 256, 128))), critic_hidden=pm.get("critic_hidden"))).to(cfg.device)
ac.load_state_dict(torch.load(run / "model.pt", map_location=cfg.device)["ac"]); ac.eval()
n, dev, e = a.n_envs, cfg.device, env.env
obs, _ = env.compute_obs()
seen = torch.zeros(n, dtype=torch.bool, device=dev)
Ko = cfg.opp_obs; flag_hist = torch.zeros(n, 30, device=dev)
rows = []
with torch.no_grad():
    for it in range(4500):
        pre = dict(pos=e.quad.pos.clone(), vel=e.quad.vel.clone(), R=quat_to_rot(e.quad.quat), opos=e.opp.pos.clone(), ovel=e.opp.vel.clone(),
                   act=e.opp.active.clone(), gi=e.gate_idx.clone(), st=e.steps.clone(), cnt=e.opp.count.clone(), flag=flag_hist.clone())
        act = ac.actor(ac.pn(obs))
        obs, _, _, _, done, info = env.step(act)
        fl = obs[:, -Ko * 7::7][:, :Ko].amax(-1) if Ko else torch.zeros(n, device=dev)
        flag_hist = torch.cat((flag_hist[:, 1:], fl[:, None]), 1)
        new = done & ~seen
        hit = new & info["opp_hit"]
        for i in hit.nonzero().flatten().tolist():
            d = (pre["opos"][i] - pre["pos"][i]).norm(dim=-1); d = torch.where(pre["act"][i], d, torch.full_like(d, 1e3)); k = int(d.argmin())
            rel = pre["opos"][i, k] - pre["pos"][i]; relb = pre["R"][i].T @ rel
            u = rel / rel.norm().clamp_min(1e-6); closing = float(((pre["vel"][i] - pre["ovel"][i, k]) * u).sum())
            rows.append(dict(t=float(pre["st"][i]) * 0.01, gate=int(pre["gi"][i]), x=float(relb[0]), y=float(relb[1]), z=float(relb[2]), dist=float(rel.norm()), closing=closing,
                             vo=float(pre["ovel"][i, k].norm()), vs=float(pre["vel"][i].norm()), n_opp=int(pre["cnt"][i]), seen=float(pre["flag"][i].mean())))
        seen |= new
        if seen.all(): break
fin = float(info["finished"].float().mean())
print(f"opponent contacts: {len(rows)} of {n} flights ({100*len(rows)/n:.1f} %)")
if rows:
    import statistics as S
    def q(k): v = sorted(r[k] for r in rows); return f"median {S.median(v):6.2f}  p10 {v[len(v)//10]:6.2f}  p90 {v[9*len(v)//10]:6.2f}"
    for k in ("t", "dist", "closing", "vs", "vo", "x", "y", "z", "seen"): print(f"  {k:8s} {q(k)}")
    print("  opponent AHEAD of us (x>0):", sum(r["x"] > 0 for r in rows), "| BEHIND (x<0):", sum(r["x"] < 0 for r in rows))
    print("  closing > 0 (we overtake / close in):", sum(r["closing"] > 0 for r in rows), "| closing <= 0 (it catches us):", sum(r["closing"] <= 0 for r in rows))
    print("  not seen in the last 0.3 s (seen<0.2):", sum(r["seen"] < 0.2 for r in rows), "| seen:", sum(r["seen"] >= 0.2 for r in rows))
    print("  episode time < 3 s (start phase):", sum(r["t"] < 3 for r in rows))
    from collections import Counter
    print("  by number of opponents:", dict(Counter(r["n_opp"] for r in rows)))
    print("  gate index histogram:", dict(sorted(Counter(r["gate"] for r in rows).items())))
