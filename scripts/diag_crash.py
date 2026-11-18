"""Why does the policy crash? Ablate noise / vehicle randomization and break crashes down by cause."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_state import evaluate  # noqa: E402

run = Path(sys.argv[1])
NAMES = {1: "gate frame", 2: "pole", 4: "ground", 5: "out of bounds/hall wall"}


def sensors_off(cfg):
    cfg.sensors.scale = 0.0


def report(label, r):
    m = (r["fin"] > 0) | (r["crash"] > 0)
    n = int(m.sum())
    crash = r["crash"] > 0
    print(f"{label:34s} | n {n:4d} | success {r['fin'][m].mean()*100:5.1f}% | crash {crash.sum()/n*100:5.1f}% | speed@crash {r['vcrash'][crash].mean():4.1f} m/s", end="")
    for c, nm in NAMES.items():
        print(f" | {nm.split('/')[0]} {((r['cause'] == c) & crash).sum()/max(1,int(crash.sum()))*100:3.0f}%", end="")
    print(flush=True)


base = dict(hall_prob=0.8, drag_prob=0.25)
kw = dict(n_envs=2048, max_iters=6000)
report("base (noise + DR, hall mix)", evaluate(run, overrides=base, **kw))
report("sensor noise OFF", evaluate(run, overrides=base, cfg_hook=sensors_off, **kw))
report("vehicle DR x0.0 (nominal drone)", evaluate(run, dr_scale=0.0, overrides=base, **kw))
report("noise OFF + nominal drone", evaluate(run, dr_scale=0.0, overrides=base, cfg_hook=sensors_off, **kw))
report("old short tracks, noise + DR", evaluate(run, overrides=dict(hall_prob=0.0, drag_prob=0.0), **kw))
