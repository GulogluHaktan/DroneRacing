"""Debug: does GateNet find gates on live flight frames (raw and augmented)?"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="runs/t3")
parser.add_argument("--gatenet", default="runs_host/g1/gatenet.pt")
parser.add_argument("--out", default="runs/flight/dbg")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.estimator import det_features  # noqa: E402
from droneracing.gatenet import GateNet, decode  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

root = Path("/workspace/DroneRacing")
meta = json.loads((root / args_cli.task / "config.json").read_text())
N = 24
cfg = make_cfg(dict(meta["args"]), N, policy_sees_params=meta["env"]["policy_sees_params"])
rig = VisionRig(N, 192, 256, hfov_deg=130.0, cam_tilt=cfg.cam_tilt)
sim, scene = build_scene(N, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=rig.attach, ground_size=150.0 * (int(N ** 0.5) + 2))
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=1, quad_factory=IsaacQuad.factory(sim, scene))
dev = env.device
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
ac.load_state_dict(torch.load(root / args_cli.task / "model.pt", map_location=dev)["ac"])
ac.eval()
_ck = torch.load(root / args_cli.gatenet, map_location=dev)
gn = GateNet(refine=_ck.get("refine", False)).to(dev)
gn.load_state_dict(_ck["net"])
gn.eval()
rig.update(env)
obs, _ = env.compute_obs()
dt = cfg.dt * cfg.decimation
out = root / args_cli.out
out.mkdir(parents=True, exist_ok=True)
from PIL import Image  # noqa: E402

with torch.no_grad():
    for step in range(60):
        rgb = rig.render(dt)
        if step in (10, 30, 50):
            aug = realistic_augment(rgb, 1.0, flow_px=torch.zeros(N, 2, device=dev))
            for name, im in (("raw", rgb), ("aug", aug)):
                logit, off = gn(im)
                d = decode(logit, off)
                ft = det_features(d, 192, 256, dev)
                print(step, name, "dets per env", [len(x) for x in d], "feat nonzero", int((ft != 0).sum()), "d0 sample", d[0][:1].tolist(), flush=True)
            Image.fromarray((torch.cat((rgb[0], aug[0]), -1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out / f"s{step}.png")
        obs, _, _, _, done, _ = env.step(ac.actor(ac.pn(obs)))
        rig.update(env)
app.close()
