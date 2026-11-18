import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#2b6cb0", "grey": "#a0aec0", "red": "#c53030", "green": "#2f855a", "orange": "#dd6b20"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 1. control-rate sweep (torch, hall mix)
names = ["50 Hz\n(t19)", "100 Hz\n(h100)", "200 Hz\n(h200)"]
succ, spd = [44.4, 57.8, 53.0], [15.2, 15.7, 15.7]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
ax[0].bar(names, succ, color=[C["grey"], C["blue"], C["grey"]]); ax[0].set_ylabel("success [%]"); ax[0].set_ylim(0, 100)
ax[0].axhline(70, color=C["red"], ls="--", lw=1); ax[0].text(2.45, 71.5, "70 % requirement", color=C["red"], ha="right", fontsize=7)
for i, v in enumerate(succ): ax[0].text(i, v + 1.5, f"{v:.1f}", ha="center")
ax[1].bar(names, spd, color=[C["grey"], C["blue"], C["grey"]]); ax[1].set_ylabel("mean path speed [m/s]"); ax[1].set_ylim(0, 24)
ax[1].axhline(20, color=C["red"], ls="--", lw=1); ax[1].text(2.45, 20.5, "20 m/s target", color=C["red"], ha="right", fontsize=7)
for i, v in enumerate(spd): ax[1].text(i, v + 0.4, f"{v:.1f}", ha="center")
plt.tight_layout(); plt.savefig("fig_rate.png", dpi=200); plt.close()

# 2. error-source ablation (policy h100, hall mix, 2048 episodes)
lab = ["baseline", "wind/gust/sag", "map error\n(unseen gates)", "attitude\nerror", "latency +\ndropout", "velocity\nerror", "gate position\nnoise (PnP)", "all gate/pose\nterms"]
val = [58.3, 60.0, 59.2, 60.3, 64.1, 65.3, 77.2, 78.3]
col = [C["grey"]] * 5 + [C["blue"], C["red"], C["red"]]
fig, ax = plt.subplots(figsize=(6.4, 2.7))
ax.bar(lab, val, color=col); ax.set_ylabel("success [%] when switched off"); ax.set_ylim(40, 90)
for i, v in enumerate(val): ax.text(i, v + 0.8, f"{v:.1f}", ha="center")
plt.xticks(fontsize=7); plt.tight_layout(); plt.savefig("fig_ablation.png", dpi=200); plt.close()

# 3. perfect vs realistic sensing
lab = ["realistic sensing\n(EKF-grade error model)", "perfect sensing\n(error model off)"]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.bar(lab, [59.1, 89.3], color=[C["grey"], C["green"]]); ax.set_ylim(0, 100); ax.set_ylabel("success [%]")
ax.axhspan(90, 94, color=C["blue"], alpha=0.25); ax.text(0.5, 95.5, "MonoRace sim: 90-94 %", ha="center", fontsize=7, color=C["blue"])
for i, v in enumerate([59.1, 89.3]): ax.text(i, v - 7, f"{v:.1f}", ha="center", color="white", fontweight="bold")
plt.xticks(fontsize=7); plt.tight_layout(); plt.savefig("fig_perfect.png", dpi=200); plt.close()
print("ok")

# 4. measured closed-loop gate-position error (EKF + GateNet, Isaac, policy h100, 24 drones, ~81k steps) vs the sim profile
import numpy as np
d = ["0-3", "3-6", "6-10", "10-15", "15-30"]
mean = [0.21, 0.12, 0.16, 0.16, 0.28]; med = [0.12, 0.04, 0.06, 0.07, 0.15]; p90 = [0.48, 0.30, 0.29, 0.30, 0.57]
prof = [0.13, 0.21, 0.31, 0.44, 0.72]
x = np.arange(len(d)); w = 0.26
fig, ax = plt.subplots(figsize=(6.4, 2.7))
ax.bar(x - w, med, w, color=C["green"], label="measured median"); ax.bar(x, mean, w, color=C["blue"], label="measured mean"); ax.bar(x + w, p90, w, color=C["orange"], label="measured 90th percentile")
ax.plot(x, prof, "k--o", ms=3, lw=1, label="error level assumed in training")
ax.axhline(0.5, color=C["red"], lw=1, ls=":"); ax.text(4.45, 0.52, "~margin left by a 1.5 m opening", color=C["red"], ha="right", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([t + " m" for t in d]); ax.set_xlabel("distance to the next gate"); ax.set_ylabel("in-plane error [m]")
ax.legend(fontsize=7, frameon=False, loc="upper left"); plt.tight_layout(); plt.savefig("fig_gate_err.png", dpi=200); plt.close()

# 5. corner visibility and detection rate vs distance to the next gate (recorded closed-loop flights, h100, 72 drones)
b = ["0-1.5 m", "1.5-3 m", "3-6 m", "6-10 m"]
c4, c23, c01 = [11, 58, 86, 96], [57, 34, 7, 1], [32, 8, 7, 3]
d4, d23 = [69, 90, 86, 94], [14, 49, 43, 33]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
x = np.arange(4)
ax[0].bar(x, c4, color=C["green"], label="all 4 corners in image"); ax[0].bar(x, c23, bottom=c4, color=C["orange"], label="2-3 corners"); ax[0].bar(x, c01, bottom=np.array(c4) + np.array(c23), color=C["red"], label="< 2 corners")
ax[0].set_xticks(x); ax[0].set_xticklabels(b, fontsize=7); ax[0].set_ylabel("share of time steps [%]"); ax[0].legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
ax[1].plot(x, d4, "-o", color=C["green"], label="4 corners in image"); ax[1].plot(x, d23, "-o", color=C["orange"], label="2-3 corners in image"); ax[1].plot(x, [4, 3, 1, 0], "-o", color=C["red"], label="< 2 corners")
ax[1].set_xticks(x); ax[1].set_xticklabels(b, fontsize=7); ax[1].set_ylabel("gate detected by GateNet [%]"); ax[1].set_ylim(0, 100); ax[1].legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
plt.tight_layout(); plt.savefig("fig_visibility.png", dpi=200); plt.close()
