"""Share of policy steps with at least one opponent inside the camera field of view and within range (surrogate env, NOT RUN YET).
Gives the real-ratio input for scripts/compare_opp_detector.py --ratios.   usage: python scripts/measure_opp_ratio.py --run runs/o1"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing import RaceEnv
from droneracing.config import make_cfg
from droneracing.ekf_loop import EkfLoopEnv
from droneracing.ppo import ActorCritic, PPOCfg

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n-envs", type=int, default=1024)
ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--rng", type=float, default=15.0)
a = ap.parse_args()
run = Path(a.run); meta = json.loads((run / "config.json").read_text()); args = dict(meta["args"])
args.update(dict(fixed_prob=1.0, fixed_jitter=[0.3, 0.05], opp_prob=1.0, ekf_loop=True))
cfg = make_cfg(args, a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"]); cfg.mid_init_prob = 0.0
env = EkfLoopEnv(RaceEnv(cfg, seed=5)); e = env.env
pm = meta.get("ppo", {})
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg(hidden=tuple(pm.get("hidden", (256, 256, 128))), critic_hidden=pm.get("critic_hidden"))).to(cfg.device)
ac.load_state_dict(torch.load(run / "model.pt", map_location=cfg.device)["ac"]); ac.eval()
obs, _ = env.compute_obs()
Ko = cfg.opp_obs; inview = 0.0; alive = 0.0; near = 0.0
with torch.no_grad():
    for _ in range(a.steps):
        obs, *_ = env.step(ac.actor(ac.pn(obs)))
        flag = obs[:, -Ko * 7::7][:, :Ko].amax(-1) > 0.5  # an opponent inside the field of view and the 25 m range
        d = e.opp.dist().min(-1).values
        inview += float(flag.float().sum()); near += float(((d < a.rng) & flag).float().sum()); alive += a.n_envs
print(f"steps with an opponent in view (<= {cfg.opp_range:.0f} m): {100*inview/alive:.1f} %  | within {a.rng:.0f} m: {100*near/alive:.1f} %  (opponents in every episode, 1-3 per episode)")
