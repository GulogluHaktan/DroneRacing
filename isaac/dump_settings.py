"""Print Kit/RTX settings that mention streaming or Fabric transforms (render-state debugging)."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import carb  # noqa: E402

st = carb.settings.get_settings()


def walk(path, d, out):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(f"{path}/{k}", v, out)
    else:
        out.append((path, d))


out = []
walk("", st.get("/"), out)
for p, v in sorted(out):
    pl = p.lower()
    if ("stream" in pl and ("rtx" in pl or "hydra" in pl or "geom" in pl)) or "fabric" in pl and ("transform" in pl or "hydra" in pl):
        print(f"SET {p} = {v}")
app.close()
