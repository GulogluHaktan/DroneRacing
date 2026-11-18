"""CPU SMOKE TEST of the DAgger pipeline with the analytic camera. Real training: isaac/train_student_isaac.py.

Run: PYTHONPATH=. python scripts/train_student.py --teacher runs/t1 --iters 200 --out runs/s1
``--teacher random`` uses an untrained actor (pipeline smoke test only).
Loop per iteration: roll out T steps with the student (teacher takes over with prob beta),
label every visited state with the teacher's action, then BPTT over the stored sequence.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from droneracing import RaceEnv  # noqa: E402
from droneracing.camera import AnalyticCamera, augment  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.student import StudentNet  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, default="random")
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta0", type=float, default=1.0, help="initial teacher-takeover probability")
    p.add_argument("--beta-iters", type=int, default=50, help="iterations to decay beta to 0")
    p.add_argument("--aug", type=float, default=1.0, help="camera augmentation severity")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/s0")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    args = {"profile": "a2rl", "markers": True, "landing": True, "fork_prob": 0.3}
    if a.teacher != "random":
        meta = json.loads((Path(a.teacher) / "config.json").read_text())
        args = dict(meta["args"])
    cfg = make_cfg(args, a.n_envs)
    cfg.device = a.device
    cfg.mid_init_prob = 0.5  # student also sees mid-track states (aerobatic gates, markers)
    env = RaceEnv(cfg, seed=a.seed)
    dev = env.device
    cam = AnalyticCamera(env)

    teacher = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
    if a.teacher != "random":
        teacher.load_state_dict(torch.load(Path(a.teacher) / "model.pt", map_location=dev)["ac"])
    teacher.eval()
    student = StudentNet().to(dev)

    out = Path(a.out)
    from droneracing.dagger import run_dagger
    run_dagger(env, cam.render, teacher, student, out, a.iters, a.T, a.lr, a.beta0, a.beta_iters, a.aug)


def quat_rot_t(q):
    from droneracing.quad import quat_to_rot
    return quat_to_rot(q).transpose(1, 2)


if __name__ == "__main__":
    main()
