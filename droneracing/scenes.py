"""Hand-authored test scenes (pure numpy, no Isaac): gates with full 3D orientation,
poles, hurdles, window walls, markers, start and landing pads.

Each scene is converted to a flat list of render primitives (box / cylinder).
Conventions match the env: gate normal n = flight direction through the gate,
right = n x z (z-up), up = right x n. ``roll`` rotates the gate about n.
Marker colours follow the task semantics (turn L/S/R, climb U/L/D, fork, finish).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BAR = 0.26  # gate frame bar thickness [m] (thick for readable previews)
DEPTH = 0.2


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def gate_basis(normal, roll: float = 0.0) -> np.ndarray:
    """Columns = (normal, right, up)."""
    n = unit(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    right = unit(np.cross(n, ref))
    up = np.cross(right, n)
    c, s = math.cos(roll), math.sin(roll)
    return np.stack((n, c * right + s * up, -s * right + c * up), axis=1)


@dataclass
class Prim:
    kind: str  # "box" | "cyl"
    center: np.ndarray
    R: np.ndarray  # 3x3, columns = local axes
    size: tuple  # box: (sx, sy, sz); cyl: (radius, height)
    color: tuple
    name: str = ""


@dataclass
class Gate:
    pos: tuple
    normal: tuple = (1.0, 0.0, 0.0)
    roll: float = 0.0
    w: float = 1.4
    h: float = 1.4
    decoy: bool = False
    marker: bool = True


@dataclass
class Scene:
    key: str
    title: str
    desc: str
    start: tuple = (0.0, 0.0, 1.0)
    start_yaw: float = 0.0
    land: tuple | None = None
    gates: list[Gate] = field(default_factory=list)
    extra: list[Prim] = field(default_factory=list)  # poles, hurdles, walls
    fork_before: dict[int, str] = field(default_factory=dict)  # gate index -> "L"/"R" (marker on previous gate)
    path: list | None = None  # explicit flight path (polyline); default = start -> gates -> land
    focus: tuple = (0.0, 0.0, 1.5)  # close-up target
    focus_dist: float = 12.0


# ---- primitive builders --------------------------------------------------------------------
def box(center, size, R=None, color=(0.7, 0.7, 0.7), name="") -> Prim:
    return Prim("box", np.asarray(center, float), np.eye(3) if R is None else R, tuple(size), color, name)


def yaw_R(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pole(x, y, r=0.15, h=3.5, color=(0.9, 0.9, 0.2)) -> Prim:
    return Prim("cyl", np.array([x, y, h / 2]), np.eye(3), (r, h), color, "pole")


def hurdle(x, y, half_span, bar_z, post_h, yaw=0.0, color=(0.95, 0.35, 0.2)) -> list[Prim]:
    R = yaw_R(yaw)
    side = R @ np.array([0.0, 1.0, 0.0])
    c = np.array([x, y, 0.0])
    out = [pole(*(c + side * s)[:2], r=0.07, h=post_h, color=color) for s in (-half_span, half_span)]
    out.append(box(c + np.array([0, 0, bar_z]), (0.08, 2 * half_span, 0.1), R, color, "bar"))
    return out


def window_wall(center, yaw, wall_w, wall_h, win_w, win_h, win_z, color=(0.55, 0.6, 0.7)) -> list[Prim]:
    """Wall (thickness 0.2) in the plane facing along yaw, with a rectangular window."""
    R = yaw_R(yaw)
    right = R @ np.array([0.0, 1.0, 0.0])
    c = np.array(center, float)
    t = 0.2
    parts = []
    left_w = (wall_w - win_w) / 2
    for s in (-1, 1):
        cc = c + right * s * (win_w / 2 + left_w / 2)
        parts.append(box(cc + np.array([0, 0, wall_h / 2]), (t, left_w, wall_h), R, color, "wall"))
    top_h = wall_h - (win_z + win_h / 2)
    parts.append(box(c + np.array([0, 0, win_z + win_h / 2 + top_h / 2]), (t, win_w, top_h), R, color, "wall"))
    bot_h = win_z - win_h / 2
    parts.append(box(c + np.array([0, 0, bot_h / 2]), (t, win_w, bot_h), R, color, "wall"))
    return parts


def gate_prims(g: Gate, color, idx: int) -> list[Prim]:
    B = gate_basis(g.normal, g.roll)
    n, r, u = B[:, 0], B[:, 1], B[:, 2]
    p = np.asarray(g.pos, float)
    return [
        box(p + u * (g.h / 2 + BAR / 2), (DEPTH, g.w + 2 * BAR, BAR), B, color, f"g{idx}_top"),
        box(p - u * (g.h / 2 + BAR / 2), (DEPTH, g.w + 2 * BAR, BAR), B, color, f"g{idx}_bot"),
        box(p + r * (g.w / 2 + BAR / 2), (DEPTH, BAR, g.h), B, color, f"g{idx}_r"),
        box(p - r * (g.w / 2 + BAR / 2), (DEPTH, BAR, g.h), B, color, f"g{idx}_l"),
    ]


def _heading(n) -> float | None:
    n = np.asarray(n, float)
    return None if abs(n[2]) > 0.8 else math.atan2(n[1], n[0])


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def marker_prims(scene: Scene, real: list[int]) -> list[Prim]:
    """Panels above each real gate: turn colour + climb stripe (+ fork / finish)."""
    out: list[Prim] = []
    gates = scene.gates
    turn_col = {"L": (0.1, 0.45, 1.0), "S": (0.1, 0.85, 0.3), "R": (1.0, 0.55, 0.1)}
    climb_col = {"U": (1, 1, 1), "L": (0.5, 0.5, 0.5), "D": (0.05, 0.05, 0.05)}
    for k, gi in enumerate(real):
        g = gates[gi]
        if not g.marker:
            continue
        last = k == len(real) - 1
        p = np.asarray(g.pos, float)
        if last:
            tc, cc = (1.0, 0.1, 0.8), (1, 1, 1)
        else:
            g2 = gates[real[k + 1]]
            h1, h2 = _heading(g.normal), _heading(g2.normal)
            dpsi = 0.0 if (h1 is None or h2 is None) else _wrap(h2 - h1)
            dz = g2.pos[2] - g.pos[2]
            t = "L" if dpsi > 0.35 else ("R" if dpsi < -0.35 else "S")
            c = "U" if dz > 0.6 else ("D" if dz < -0.6 else "L")
            tc, cc = turn_col[t], climb_col[c]
        hdg = _heading(g.normal)
        R = yaw_R(hdg if hdg is not None else 0.0)
        top = p + np.array([0, 0, max(g.w, g.h) / 2 + BAR + 0.55])
        out.append(box(top, (0.05, 0.55, 0.45), R, tc, f"m{k}_turn"))
        out.append(box(top - np.array([0, 0, 0.32]), (0.05, 0.55, 0.12), R, cc, f"m{k}_climb"))
        if (k + 1) in scene.fork_before and k + 1 < len(real) + 1:
            pass
    for gi, side in scene.fork_before.items():
        # marker on the gate before the fork gate: purple panel with a white square on the correct side
        k = real.index(gi) - 1
        if k < 0:
            continue
        g = gates[real[k]]
        hdg = _heading(g.normal) or 0.0
        R = yaw_R(hdg)
        right = R @ np.array([0, 1.0, 0])
        p = np.asarray(g.pos, float) + np.array([0, 0, max(g.w, g.h) / 2 + BAR + 1.35])
        out.append(box(p, (0.05, 0.7, 0.4), R, (0.55, 0.2, 0.8), "fork_panel"))
        s = 1.0 if side == "L" else -1.0  # left of travel = -right
        out.append(box(p - right * s * 0.2 + R @ np.array([0.04, 0, 0]), (0.05, 0.22, 0.22), R, (1, 1, 1), "fork_side"))
    return out


def gradient(i: int, n: int) -> tuple:
    t = i / max(n - 1, 1)
    return (0.1 + 0.9 * t, 0.85 - 0.6 * t, 1.0 - 0.7 * t)


def drone_prims(pos, yaw) -> list[Prim]:
    R = yaw_R(yaw)
    p = np.asarray(pos, float)
    out = [box(p, (0.16, 0.16, 0.06), R, (0.1, 0.1, 0.1), "drone_body")]
    for ang in (math.pi / 4, -math.pi / 4):
        out.append(box(p, (0.5, 0.035, 0.02), yaw_R(yaw + ang), (0.15, 0.15, 0.15), "arm"))
    for sx in (-1, 1):
        for sy in (-1, 1):
            off = R @ np.array([sx * 0.18, sy * 0.18, 0.03])
            out.append(Prim("cyl", p + off, np.eye(3), (0.09, 0.015), (0.2, 0.9, 0.3), "rotor"))
    return out


def scene_prims(scene: Scene, offset=(0.0, 0.0, 0.0)) -> list[Prim]:
    off = np.asarray(offset, float)
    real = [i for i, g in enumerate(scene.gates) if not g.decoy]
    out: list[Prim] = []
    for k, gi in enumerate(real):
        out += gate_prims(scene.gates[gi], gradient(k, len(real)), k)
    for gi, g in enumerate(scene.gates):
        if g.decoy:
            out += gate_prims(g, (0.85, 0.1, 0.1), 99)
    out += marker_prims(scene, real)
    out += scene.extra
    # start pad + drone
    s = np.asarray(scene.start, float)
    out.append(box([s[0], s[1], 0.025], (1.4, 1.4, 0.05), yaw_R(scene.start_yaw), (0.25, 0.3, 0.6), "start_pad"))
    out += drone_prims(s - np.array([0, 0, s[2] - 0.1]), scene.start_yaw)
    if scene.land is not None:
        l = np.asarray(scene.land, float)
        out.append(Prim("cyl", np.array([l[0], l[1], 0.02]), np.eye(3), (1.5, 0.04), (0.95, 0.95, 0.95), "land_pad"))
        out.append(Prim("cyl", np.array([l[0], l[1], 0.045]), np.eye(3), (0.8, 0.04), (0.9, 0.1, 0.1), "land_pad_in"))
    # flight path polyline (+ direction chevrons)
    if scene.path is not None:
        pts = [np.asarray(q, float) for q in scene.path]
    else:
        pts = [np.asarray(scene.start, float)] + [np.asarray(scene.gates[i].pos, float) for i in real]
        if scene.land is not None:
            pts.append(np.asarray(scene.land, float) + np.array([0, 0, 0.3]))
    since = 2.5  # path length since the last chevron (first one lands at 2.5 m)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        x = d / L
        ref = np.array([0, 0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0, 0])
        y = unit(np.cross(ref, x))
        z = np.cross(x, y)
        out.append(box((a + b) / 2, (L, 0.07, 0.07), np.stack((x, y, z), 1), (1.0, 0.9, 0.1), "path"))
        t = max(0.0, 5.0 - since)
        last_t = None
        while t < L and not __import__("os").environ.get("NOARROW"):  # chevron every ~5 m pointing along travel
            tip = a + x * t
            for sgn in (-1, 1):
                ad = unit(-math.cos(0.55) * x + sgn * math.sin(0.55) * y)
                ay = unit(np.cross(np.array([0, 0, 1.0]), ad)) if abs(ad[2]) < 0.95 else y
                az = np.cross(ad, ay)
                out.append(box(tip + ad * 0.3, (0.6, 0.12, 0.12), np.stack((ad, ay, az), 1), (1.0, 0.45, 0.0), "arrow"))
            last_t = t
            t += 5.0
        since = since + L if last_t is None else L - last_t
    for pr in out:
        pr.center = pr.center + off
    return out


# ---- the five test scenes -------------------------------------------------------------------
def _heading_n(deg: float) -> tuple:
    a = math.radians(deg)
    return (math.cos(a), math.sin(a), 0.0)


def scene_slalom_fork() -> Scene:
    s = Scene("s1_slalom_fork", "1 - Slalom + fork + poles",
              "3 slalom gates, fork (correct gate vs red decoy), final gate, landing pad; poles near the line.")
    for p, h in (((9, 0, 1.5), 0), ((20, 3.5, 1.8), 15), ((31, -1.0, 1.5), -20)):
        s.gates.append(Gate(p, _heading_n(h)))
    s.gates.append(Gate((43, -2.25, 1.5), _heading_n(0)))  # correct (left)
    s.gates.append(Gate((43, -5.25, 1.5), _heading_n(0), decoy=True))
    s.fork_before[3] = "L"
    s.gates.append(Gate((55, 0.0, 2.0), _heading_n(10)))
    s.land = (63.0, 0.5, 0.0)
    s.focus, s.focus_dist = (40.0, -3.0, 1.8), 16.0
    for x, y in ((14, 0.0), (26, 2.4), (37, -4.0), (49, 1.6)):
        s.extra.append(pole(x, y))
    return s


def scene_ladder() -> Scene:
    s = Scene("s2_ladder", "2 - Ladder (stacked gates) up and dive down",
              "Entry gate, 2 stacked horizontal gates (fly straight up), 2 stacked (dive), exit, landing.")
    s.gates.append(Gate((7, 0, 1.5), (1, 0, 0)))
    for z in (3.0, 5.2):
        s.gates.append(Gate((14, 0, z), (0, 0, 1), w=1.6, h=1.6))
    for z in (5.2, 3.0):
        s.gates.append(Gate((20, 0, z), (0, 0, -1), w=1.6, h=1.6))
    s.gates.append(Gate((28, 0, 1.3), (1, 0, 0)))
    s.land = (35.0, 0.0, 0.0)
    s.path = [(0, 0, 1.0), (7, 0, 1.5), (14, 0, 1.2), (14, 0, 3.0), (14, 0, 5.2), (17, 0, 6.6), (20, 0, 5.2),
              (20, 0, 3.0), (20, 0, 1.2), (28, 0, 1.3), (35, 0, 0.3)]
    s.focus, s.focus_dist = (17.0, 0.0, 4.0), 16.0
    return s


def scene_power_loop() -> Scene:
    s = Scene("s3_power_loop", "3 - Stacked loop (bottom -> top inverted -> bottom)",
              "Two gates stacked: fly through the lower one, loop over, pass the upper one inverted (reverse "
              "direction), come back through the lower one and continue.")
    cx, R, zb = 18.0, 2.5, 1.5
    s.gates.append(Gate((8, 0, 1.5), (1, 0, 0)))
    s.gates.append(Gate((cx, 0, zb), (1, 0, 0), w=1.6, h=1.4, marker=False))  # lower
    s.gates.append(Gate((cx, 0, zb + 2 * R), (-1, 0, 0), w=1.6, h=1.4, marker=False))  # upper, passed inverted
    s.gates.append(Gate((30, 0, 1.6), (1, 0, 0)))
    s.land = (38.0, 0.0, 0.0)
    loop = [(cx + R * math.sin(t), 0.0, zb + R - R * math.cos(t)) for t in np.linspace(0, 2 * math.pi, 41)]
    s.path = [(0, 0, 1.0), (8, 0, 1.5)] + loop + [(30, 0, 1.6), (38, 0, 0.3)]
    s.focus, s.focus_dist = (cx, 0.0, zb + R), 15.0
    return s


def scene_knife_corkscrew() -> Scene:
    s = Scene("s4_knife_corkscrew", "4 - Knife-edge slits + corkscrew (roll)",
              "Two narrow vertical slits (need ~90 deg bank), then 3 gates on a helix with rotating slot (barrel roll).")
    s.gates.append(Gate((10, 0.4, 1.6), (1, 0, 0), w=0.55, h=1.8))
    s.gates.append(Gate((21, -0.4, 1.6), (1, 0, 0), w=0.55, h=1.8))
    R = 1.6
    for k in range(3):
        phi = math.radians(120 * k)
        s.gates.append(Gate((32 + 5.5 * k, R * math.cos(phi), 3.2 + R * math.sin(phi)), (1, 0, 0), roll=phi, w=1.1, h=1.9))
    s.gates.append(Gate((50, 0, 1.6), (1, 0, 0)))
    s.land = (58.0, 0.0, 0.0)
    s.focus, s.focus_dist = (36.0, 0.0, 3.0), 16.0
    return s


def scene_obstacle_mix() -> Scene:
    s = Scene("s5_obstacle_mix", "5 - Obstacle mix (window, hurdle, slalom, tunnel)",
              "Window in a wall, bar to pass under, pole slalom with a gate, 2-gate tunnel, landing pad.")
    s.extra += window_wall((10, 0, 0), 0.0, 7.0, 5.0, 1.6, 1.4, 1.8)
    s.gates.append(Gate((10, 0, 1.8), (1, 0, 0), w=1.6, h=1.4))  # marks the window opening
    s.extra += hurdle(21, 0, 2.5, bar_z=1.7, post_h=2.2)
    s.gates.append(Gate((26, 0, 0.9), (1, 0, 0), w=1.6, h=1.0))  # low gate after the bar
    for i, x in enumerate((31, 36, 41, 46)):
        s.extra.append(pole(x, 1.1 if i % 2 == 0 else -1.1))
    s.gates.append(Gate((38.5, 0, 1.6), (1, 0, 0)))
    for k in range(2):
        s.gates.append(Gate((54 + 4.0 * k, 0, 1.5), (1, 0, 0), w=1.2, h=1.2))
    s.land = (66.0, 0.0, 0.0)
    s.focus, s.focus_dist = (15.0, 0.0, 1.5), 14.0
    return s


ALL_SCENES = [scene_slalom_fork, scene_ladder, scene_power_loop, scene_knife_corkscrew, scene_obstacle_mix]
