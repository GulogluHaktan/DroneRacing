"""Weight interpolation of two policies of the same lineage (same observation layout).
usage: ~/isaac-env/bin/python -I scripts/make_soup.py <safe run dir> <fast run dir> <alpha of the fast policy> <out run dir>
soup85 = (1 - 0.85) * runs/g37 + 0.85 * runs/g38d:  scripts/make_soup.py runs/g37 runs/g38d 0.85 runs/soup85
The config.json of the fast policy is copied; log_std is interpolated too (all float tensors of the actor-critic)."""
import os
import shutil
import sys

import torch

a_dir, b_dir, alpha, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
a = torch.load(f"{a_dir}/model.pt", map_location="cpu")
b = torch.load(f"{b_dir}/model.pt", map_location="cpu")
assert set(a["ac"]) == set(b["ac"]), "the two policies do not share one parameter layout"
sd = {k: ((1 - alpha) * a["ac"][k] + alpha * b["ac"][k]) if a["ac"][k].is_floating_point() else b["ac"][k] for k in a["ac"]}
os.makedirs(out, exist_ok=True)
torch.save({"ac": sd, "it": 0}, f"{out}/model.pt")
shutil.copy(f"{b_dir}/config.json", f"{out}/config.json")
print("wrote", out, f"= {1 - alpha:.2f} * {a_dir} + {alpha:.2f} * {b_dir}")
