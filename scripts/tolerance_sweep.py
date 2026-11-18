"""How much perception error does a policy tolerate? Torch-env success while scaling gate / velocity estimate errors."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_state as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n-envs", type=int, default=2048)
a = ap.parse_args()
orig = E.make_cfg
mods = {}


def wrapped(args, n, **kw):
    cfg = orig(args, n, **kw)
    s = cfg.sensors
    g, v, drift, prof = mods["g"], mods["v"], mods["drift"], mods.get("prof")
    if prof == "closedloop":
        cfg2 = orig({**args, "sensor_profile": "closedloop"}, n, **kw)
        cfg.sensors = cfg2.sensors
        return cfg
    s.gate_noise *= g; s.gate_noise_dist *= g; s.gate_bias *= g; s.gate_yaw_bias *= g
    s.vel_noise *= v; s.vel_bias *= v; s.vel_drift = drift
    return cfg


E.make_cfg = wrapped
for name, m in (("as trained (vision profile)", dict(g=1, v=1, drift=0.0)),
                ("gate error x0 (perfect gates)", dict(g=0, v=1, drift=0.0)),
                ("gate error x2", dict(g=2, v=1, drift=0.0)),
                ("gate error x4", dict(g=4, v=1, drift=0.0)),
                ("gate error x8", dict(g=8, v=1, drift=0.0)),
                ("vel drift 0.45 m/s/sqrt(s)", dict(g=1, v=1, drift=0.45)),
                ("vel drift 0.9", dict(g=1, v=1, drift=0.9)),
                ("vel err x3 (no drift)", dict(g=1, v=3, drift=0.0)),
                ("MEASURED closed-loop profile", dict(g=1, v=1, drift=0.0, prof="closedloop"))):
    mods.clear(); mods.update(m)
    r = E.evaluate(Path(a.run), a.n_envs, 12345)
    E.summarize(r, name[:30])
