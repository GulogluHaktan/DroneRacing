"""Which part of the EKF-grade sensor model costs success? Switch one error group off at a time."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_state import evaluate  # noqa: E402

run = Path(sys.argv[1])
base = dict(hall_prob=0.8, drag_prob=0.25)


def zero(*names):
    def hook(cfg):
        for n in names:
            v = getattr(cfg.sensors, n)
            setattr(cfg.sensors, n, (0, 0) if isinstance(v, tuple) else 0.0)
    return hook


groups = {
    "baseline": zero(),
    "pose error OU (odo_sigma)": zero("odo_sigma"),
    "gate PnP noise (gate_noise*)": zero("gate_noise", "gate_noise_dist"),
    "map error unseen (gate_bias, yaw_bias)": zero("gate_bias", "gate_yaw_bias"),
    "gate-normal noise (yaw_noise)": zero("yaw_noise"),
    "ALL gate/pose terms": zero("odo_sigma", "gate_noise", "gate_noise_dist", "gate_bias", "gate_yaw_bias", "yaw_noise"),
    "velocity error": zero("vel_noise", "vel_bias", "vel_noise_rel", "vel_drift"),
    "attitude error": zero("att_noise", "att_bias"),
    "latency + dropout": zero("obs_delay", "act_delay", "gate_dropout"),
    "wind/gust/sag": zero("wind", "gust", "sag"),
}
for name, hook in groups.items():
    r = evaluate(run, n_envs=2048, max_iters=6000, overrides=base, cfg_hook=hook)
    m = (r["fin"] > 0) | (r["crash"] > 0)
    c = r["crash"] > 0
    print(f"{name:42s} | success {r['fin'][m].mean()*100:5.1f}% | speed@crash {r['vcrash'][c].mean():4.1f}", flush=True)
