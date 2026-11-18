"""Rule-based procedural track generator, vectorized over environments.

Rules (see TrackRules): gate count, segment length, bounded heading change
(scaled by vehicle capability), bounded climb angle, height band, clearance
between non-adjacent gates, per-track gate size, and vertical-pole obstacles
placed near the straight gate-to-gate line with spacing constraints.

Marker semantics (used when EnvCfg.markers is on):
* every gate carries a *direction marker* about the NEXT segment: turn in {L,S,R}
  and climb in {D,L,U} (the last gate carries FINISH);
* with probability fork_prob one gate is a *fork*: the correct gate and a decoy side by
  side; the marker on the PREVIOUS gate says which side (L/R) is correct;
* a landing point sits land_dist after the last gate (slot index n_gates).
Tensors have G+1 gate slots; slot n_gates holds the landing point.
"""
from __future__ import annotations

import math

import torch

from .config import TrackRules

TURN_L, TURN_S, TURN_R, TURN_U = 0, 1, 2, 3
CLIMB_D, CLIMB_L, CLIMB_U = 0, 1, 2
FORK_NONE, FORK_L, FORK_R = 0, 1, 2


def _u(shape, lo: float, hi: float, gen, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _hall_laps(n: int, rules: TrackRules, G: int, gen, device):
    """Closed laps inside a hall: a perturbed superellipse sampled at L gates, passed for 1-2 laps (identical gates).

    Returns (valid, pos [n,G+1,3], yaw [n,G+1], count, hall box). Invalid laps (gates too close, too many gates) are
    replaced by the random-walk track at the call site."""
    Lr, laps_r = rules.hall_gates_per_lap, rules.hall_laps
    L = torch.randint(Lr[0], Lr[1] + 1, (n,), generator=gen, device=device)
    laps = torch.randint(laps_r[0], laps_r[1] + 1, (n,), generator=gen, device=device)
    count = L * laps
    W, H = _u((n,), *rules.hall_w, gen, device), _u((n,), *rules.hall_h, gen, device)
    a, b = (W / 2 - rules.hall_margin).clamp_min(5.0), (H / 2 - rules.hall_margin).clamp_min(4.0)
    ps = _u((n,), *rules.hall_shape, gen, device)
    direction = torch.where(torch.rand(n, generator=gen, device=device) < 0.5, -1.0, 1.0)
    th0 = _u((n,), 0.0, 2 * math.pi, gen, device)
    Lmax = Lr[1]
    j = torch.arange(Lmax, device=device)[None].float()
    jit = _u((n, Lmax), -0.25, 0.25, gen, device) / L[:, None].float()  # fraction of a full turn
    th = th0[:, None] + direction[:, None] * 2 * math.pi * (j / L[:, None].float() + jit)
    e1, e2 = _u((n, 1), 0, rules.hall_wavy, gen, device), _u((n, 1), 0, rules.hall_wavy, gen, device)
    f1, f2 = _u((n, 1), 0, 2 * math.pi, gen, device), _u((n, 1), 0, 2 * math.pi, gen, device)
    rad = 1.0 + e1 * torch.cos(2 * th + f1) + e2 * torch.cos(3 * th + f2)
    c, s_ = torch.cos(th), torch.sin(th)
    ex = 2.0 / ps[:, None]
    x = a[:, None] * torch.sign(c) * c.abs() ** ex * rad
    y = b[:, None] * torch.sign(s_) * s_.abs() ** ex * rad
    x = x.clamp(-a[:, None], a[:, None])
    y = y.clamp(-b[:, None], b[:, None])
    z0 = _u((n, 1), rules.z_range[0] + rules.hall_z_amp + 0.2, rules.z_range[1] - rules.hall_z_amp - 0.2, gen, device)
    z = z0 + rules.hall_z_amp * torch.sin(th + _u((n, 1), 0, 2 * math.pi, gen, device)) * _u((n, 1), 0.0, 1.0, gen, device)
    P = torch.stack((x, y, z.clamp(*rules.z_range)), -1)  # [n, Lmax, 3], valid for index < L
    # gather lap points for all slots: slot k -> lap index k mod L
    k = torch.arange(G + 1, device=device)[None]
    within = (k % L[:, None]).clamp(max=Lmax - 1)
    pos = torch.gather(P, 1, within[..., None].expand(n, G + 1, 3))
    nxt = torch.gather(P, 1, ((within + 1) % L[:, None])[..., None].expand(n, G + 1, 3))
    prv = torch.gather(P, 1, ((within - 1) % L[:, None])[..., None].expand(n, G + 1, 3))
    yaw = torch.atan2(nxt[..., 1] - prv[..., 1], nxt[..., 0] - prv[..., 0])
    pos, yaw = torch.where((k < count[:, None])[..., None], pos, pos[:, :1]), yaw
    # validity: lap gates separated (non-adjacent >= clearance, adjacent >= 3 m), count fits in the slots
    dist = (P[:, :, None] - P[:, None]).norm(dim=-1)  # [n, Lmax, Lmax]
    ii = torch.arange(Lmax, device=device)
    real = (ii[None, :, None] < L[:, None, None]) & (ii[None, None, :] < L[:, None, None])
    ring = (ii[:, None] - ii[None]).abs()
    adj = (ring == 1) | (ring == (L[:, None, None] - 1))
    dmin_nonadj = torch.where(real & ~adj & (ring > 0), dist, torch.full_like(dist, 1e6)).amin(dim=(1, 2))
    dmin_adj = torch.where(real & adj & (ring > 0), dist, torch.full_like(dist, 1e6)).amin(dim=(1, 2))
    ok = (dmin_nonadj >= rules.clearance) & (dmin_adj >= 3.0) & (count <= G)
    box = torch.stack((torch.zeros(n, device=device), torch.zeros(n, device=device), a + 6.5, b + 6.5), -1)  # walls 6.5 m outside the gate region (start run-up)
    return ok, pos, yaw, count.clamp(max=G), box


def a2rl_2025_lap(device):
    """One lap of the A2RL x DCL 2025 final track (11 gates), read off the top-down plot in the KAIST paper
    (arXiv 2512.20475, Fig. c03: x 0-95 m, y 5-27 m, gate order from Fig. 1) and shifted to the 100 x 30 m hall centre.
    Gates 5/6 are the split-S (upper gate, then the lower gate beneath it, opposite direction); gate 4 is flown through the
    upper opening of a double gate. Low gate centres 1.45 m, upper 4.15 m (stacked 2.7 m frames). Positions +-1 m (plot reading)."""
    lo, hi = 1.45, 4.15
    g = [  # x, y, z, yaw [deg] (direction of passage)
        (14.0, 25.0, lo, 30.0), (30.5, 19.5, lo, -18.0), (46.0, 22.0, lo, 10.0), (63.0, 20.0, hi, 0.0),
        (85.0, 18.0, hi, 0.0), (85.0, 18.0, lo, 180.0), (68.0, 13.0, lo, -165.0), (55.0, 7.0, lo, 180.0),
        (37.0, 12.0, lo, 180.0), (19.0, 7.0, lo, 180.0), (8.0, 14.0, lo, 90.0),
    ]
    t = torch.tensor(g, device=device)
    pos = t[:, :3] - torch.tensor([50.0, 15.0, 0.0], device=device)
    return pos, torch.deg2rad(t[:, 3])


def _generate_once(n: int, rules: TrackRules, capability: torch.Tensor, gen, device) -> dict[str, torch.Tensor]:
    G, M, R = rules.max_gates, rules.max_obstacles, rules.n_retries
    gate_size = _u((n,), rules.gate_size[0], rules.gate_size[1], gen, device)
    pos = torch.zeros(n, G + 1, 3, device=device)
    yaw = torch.zeros(n, G + 1, device=device)
    wmax = rules.walk_max_gates or G
    n_target = torch.randint(rules.min_gates, wmax + 1, (n,), generator=gen, device=device)
    drag = torch.zeros(n, dtype=torch.bool, device=device)
    if rules.drag_prob > 0:  # near-straight corridor with long gaps (random draws only when enabled: old seeds keep their tracks)
        drag = torch.rand(n, generator=gen, device=device) < rules.drag_prob
        n_drag = torch.randint(rules.drag_gates[0], rules.drag_gates[1] + 1, (n,), generator=gen, device=device)
        n_target = torch.where(drag, n_drag, n_target)
    seg_lo = torch.where(drag, rules.drag_seg[0], rules.seg_len[0]) if rules.drag_prob > 0 else None
    seg_hi = torch.where(drag, rules.drag_seg[1], rules.seg_len[1]) if rules.drag_prob > 0 else None

    yaw[:, 0] = _u((n,), 0.0, 2 * math.pi, gen, device)
    pos[:, 0, 2] = _u((n,), rules.z_range[0] + 0.3, rules.z_range[1] - 0.3, gen, device)
    turn_max = rules.turn_max[0] + capability * (rules.turn_max[1] - rules.turn_max[0])
    if rules.drag_prob > 0:
        turn_max = torch.where(drag, torch.full_like(turn_max, 0.12), turn_max)

    count = torch.ones(n, dtype=torch.long, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    batch = torch.arange(n, device=device)
    tan_climb = math.tan(rules.max_climb)
    fixed = torch.zeros(n, G + 1, dtype=torch.bool, device=device)  # slots pre-assigned by a module
    for k in range(1, G):
        is_fixed = fixed[:, k] & alive
        dpsi = (torch.rand(n, R, generator=gen, device=device) * 2 - 1) * turn_max[:, None]
        if seg_lo is None:
            d = _u((n, R), rules.seg_len[0], rules.seg_len[1], gen, device)
        else:
            d = seg_lo[:, None] + (seg_hi - seg_lo)[:, None] * torch.rand(n, R, generator=gen, device=device)
        psi = yaw[:, k - 1, None] + dpsi
        dz = (torch.rand(n, R, generator=gen, device=device) * 2 - 1) * d * tan_climb
        cand = torch.stack(
            (
                pos[:, k - 1, 0, None] + d * torch.cos(psi),
                pos[:, k - 1, 1, None] + d * torch.sin(psi),
                (pos[:, k - 1, 2, None] + dz).clamp(*rules.z_range),
            ),
            dim=-1,
        )  # [n, R, 3]
        if k >= 2:
            dist = (cand[:, :, None, :] - pos[:, None, : k - 1, :]).norm(dim=-1)  # [n, R, k-1]
            valid = (dist >= rules.clearance).all(dim=-1)
        else:
            valid = torch.ones(n, R, dtype=torch.bool, device=device)
        first = valid.float().argmax(dim=-1)
        extend = alive & valid.any(dim=-1) & (k < n_target) & ~is_fixed
        new_pos = cand[batch, first]
        new_yaw = psi[batch, first]
        pos[:, k] = torch.where(extend[:, None], new_pos, torch.where(is_fixed[:, None], pos[:, k], pos[:, k - 1]))
        yaw[:, k] = torch.where(extend, new_yaw, torch.where(is_fixed, yaw[:, k], yaw[:, k - 1]))
        count += (extend | is_fixed).long()
        alive = extend | is_fixed

        # aerobatic modules start at a freshly placed ordinary gate k and pre-assign the next slots
        if k >= 2 and (rules.split_prob > 0 or rules.loop_prob > 0):
            r = torch.rand(n, generator=gen, device=device)
            split = extend & (r < rules.split_prob) & (k + 1 < G) & (k + 1 < n_target)
            loop = extend & ~split & (r > 1 - rules.loop_prob) & (k + 2 < G) & (k + 2 < n_target)
            dzs = gate_size + 2 * rules.gate_frame + _u((n,), rules.split_gap[0], rules.split_gap[1], gen, device)
            dzl = _u((n,), rules.loop_dz[0], rules.loop_dz[1], gen, device)
            pos[:, k, 2] = torch.where(loop, pos[:, k, 2].clamp(max=2.0), pos[:, k, 2])  # keep the loop top in-hall
            zlo = rules.z_range[0]
            # split-S (A2RL rule): upper gate first, then the lower gate directly beneath it in the opposite
            # direction; spacing follows the gate size, small lateral / yaw misalignment is randomized
            up_z = torch.maximum(pos[:, k, 2], zlo + dzs)
            pos[:, k, 2] = torch.where(split, up_z, pos[:, k, 2])
            if k + 1 < G:
                lat = (torch.rand(n, generator=gen, device=device) * 2 - 1) * rules.split_lat
                yj = (torch.rand(n, generator=gen, device=device) * 2 - 1) * rules.split_yaw
                right_k = torch.stack((torch.sin(yaw[:, k]), -torch.cos(yaw[:, k]), torch.zeros(n, device=device)), -1)
                lower = pos[:, k].clone() + lat[:, None] * right_k
                lower[:, 2] = up_z - dzs
                pos[:, k + 1] = torch.where(split[:, None], lower, pos[:, k + 1])
                yaw[:, k + 1] = torch.where(split, yaw[:, k] + math.pi + yj, yaw[:, k + 1])
                fixed[:, k + 1] |= split
            # loop: slot k = lower gate, k+1 = upper gate (inverted pass), k+2 = lower gate again
            if k + 2 < G:
                upper = pos[:, k].clone()
                upper[:, 2] = pos[:, k, 2] + dzl
                pos[:, k + 1] = torch.where(loop[:, None], upper, pos[:, k + 1])
                yaw[:, k + 1] = torch.where(loop, yaw[:, k] + math.pi, yaw[:, k + 1])
                pos[:, k + 2] = torch.where(loop[:, None], pos[:, k], pos[:, k + 2])
                yaw[:, k + 2] = torch.where(loop, yaw[:, k], yaw[:, k + 2])
                fixed[:, k + 1] |= loop
                fixed[:, k + 2] |= loop
        # ladder (assumed geometry): slot k + two rungs straight ahead, each a height step up (or all down)
        if k >= 2 and rules.ladder_prob > 0 and k + 2 < G:
            lad = extend & ~fixed[:, k] & (torch.rand(n, generator=gen, device=device) < rules.ladder_prob) & (k + 2 < G) & (k + 2 < n_target)
            if rules.split_prob > 0 or rules.loop_prob > 0:
                lad &= ~(split | loop)
            step = _u((n,), rules.ladder_step[0], rules.ladder_step[1], gen, device)
            down = torch.rand(n, generator=gen, device=device) < 0.5
            zlo_, zhi_ = rules.z_range
            z0 = torch.where(down, pos[:, k, 2].clamp(min=zlo_ + 2 * step), pos[:, k, 2].clamp(max=zhi_ - 2 * step)).clamp(zlo_, zhi_)
            pos[:, k, 2] = torch.where(lad, z0, pos[:, k, 2])
            fwd = torch.stack((torch.cos(yaw[:, k]), torch.sin(yaw[:, k]), torch.zeros(n, device=device)), -1)
            sgn = torch.where(down, -1.0, 1.0)
            for j in (1, 2):
                dx = _u((n,), rules.ladder_dx[0], rules.ladder_dx[1], gen, device)
                rung = pos[:, k + j - 1] + dx[:, None] * fwd
                rung[:, 2] = (pos[:, k + j - 1, 2] + sgn * step).clamp(zlo_, zhi_)
                pos[:, k + j] = torch.where(lad[:, None], rung, pos[:, k + j])
                yaw[:, k + j] = torch.where(lad, yaw[:, k], yaw[:, k + j])
                fixed[:, k + j] |= lad
    pos[:, G] = pos[:, G - 1]
    yaw[:, G] = yaw[:, G - 1]
    hall = torch.tensor([0.0, 0.0, 1e4, 1e4], device=device).repeat(n, 1)  # (cx, cy, half-length, half-width); 1e4 = no walls
    if rules.hall_prob > 0:
        h_ok, h_pos, h_yaw, h_cnt, h_box = _hall_laps(n, rules, G, gen, device)
        use = h_ok & (torch.rand(n, generator=gen, device=device) < rules.hall_prob)
        pos = torch.where(use[:, None, None], h_pos, pos)
        yaw = torch.where(use[:, None], h_yaw, yaw)
        count = torch.where(use, h_cnt, count)
        hall = torch.where(use[:, None], h_box, hall)

    if rules.fixed_prob > 0:  # replica of the A2RL x DCL 2025 final track (random draws only when enabled)
        f_pos, f_yaw = a2rl_2025_lap(device)
        L = f_pos.shape[0]
        laps = min(2, G // L)
        k = torch.arange(G + 1, device=device) % L
        jt = rules.fixed_jitter
        jp = (torch.rand(n, L, 3, generator=gen, device=device) * 2 - 1) * torch.tensor((jt[0], jt[0], 0.3 * jt[0]), device=device)
        jy = (torch.rand(n, L, generator=gen, device=device) * 2 - 1) * jt[1]
        sc = _u((n, 1, 1), *rules.fixed_scale, gen, device)  # whole-layout scale in x/y (hall clamp below)
        mir = torch.where(torch.rand(n, 1, generator=gen, device=device) < rules.fixed_mirror, -1.0, 1.0)  # mirrored layout
        fp = f_pos[None].repeat(n, 1, 1)
        fp[..., :2] = fp[..., :2] * sc
        fp[..., 1] = fp[..., 1] * mir
        fp = fp + jp
        sp = (jp[:, 4] - jp[:, 5]) * torch.tensor([1.0, 1.0, 0.0], device=device)
        fp[:, 5] = fp[:, 5] + sp  # split-S lower gate stays under the upper one (height jitter only)
        fp[..., 0] = fp[..., 0].clamp(-46.0, 46.0)
        fp[..., 1] = fp[..., 1].clamp(-13.0, 13.0)
        fy = f_yaw[None] * mir + jy
        fp, fy = fp[:, k], fy[:, k]
        use = torch.rand(n, generator=gen, device=device) < rules.fixed_prob
        pos = torch.where(use[:, None, None], fp, pos)
        yaw = torch.where(use[:, None], fy, yaw)
        count = torch.where(use, torch.full_like(count, L * laps), count)
        hall = torch.where(use[:, None], torch.tensor([0.0, 0.0, 50.0, 15.0], device=device), hall)  # 100 x 30 m hall

    # ---- markers -----------------------------------------------------------------------
    idx = torch.arange(G + 1, device=device)[None]  # [1, G+1]
    nxt = (idx + 1).clamp(max=G)
    dpsi_k = _wrap(yaw[:, nxt[0]] - yaw)  # heading change at gate k -> k+1
    dz_k = pos[:, nxt[0], 2] - pos[:, :, 2]
    is_last = idx == (count[:, None] - 1)
    turn = torch.where(dpsi_k > rules.turn_thresh, TURN_L, torch.where(dpsi_k < -rules.turn_thresh, TURN_R, TURN_S))
    climb = torch.where(dz_k > rules.climb_thresh, CLIMB_U, torch.where(dz_k < -rules.climb_thresh, CLIMB_D, CLIMB_L))
    turn = torch.where(dpsi_k.abs() > rules.turnu_thresh, TURN_U, turn)
    turn = torch.where(is_last | (idx >= count[:, None]), TURN_S, turn)
    climb = torch.where(is_last | (idx >= count[:, None]), CLIMB_L, climb)

    # ---- fork: correct gate shifted aside, decoy on the other side ----------------------
    decoy_pos = pos.clone()
    decoy_on = torch.zeros(n, G + 1, dtype=torch.bool, device=device)
    fork_side = torch.full((n, G + 1), FORK_NONE, dtype=torch.long, device=device)
    if rules.fork_prob > 0:
        has = (torch.rand(n, generator=gen, device=device) < rules.fork_prob) & (count >= 3)
        j = (1 + torch.rand(n, generator=gen, device=device) * (count - 1).clamp_min(1)).long().clamp(max=G - 1)
        j = torch.minimum(j, (count - 1).clamp_min(1))
        side = torch.where(torch.rand(n, generator=gen, device=device) < 0.5, -1.0, 1.0)  # +1: correct gate on the right
        off = _u((n,), rules.fork_offset[0], rules.fork_offset[1], gen, device)
        yj = yaw[batch, j]
        right = torch.stack((torch.sin(yj), -torch.cos(yj), torch.zeros_like(yj)), -1)
        centre = pos[batch, j]
        shift = (0.5 * off * side)[:, None] * right
        h = has[:, None]
        pos[batch, j] = torch.where(h, centre + shift, centre)
        decoy_pos[batch, j] = torch.where(h, centre - shift, centre)
        decoy_on[batch, j] = has
        fork_side[batch, j - 1] = torch.where(has, torch.where(side > 0, FORK_R, FORK_L), FORK_NONE)

    # ---- landing point after the last gate ---------------------------------------------
    y_last = yaw[batch, count - 1]
    n_last = torch.stack((torch.cos(y_last), torch.sin(y_last), torch.zeros_like(y_last)), -1)
    land = pos[batch, count - 1] + rules.land_dist * n_last
    land[:, 2] = 0.15
    pos[batch, count] = land
    yaw[batch, count] = y_last


    # ---- obstacles ---------------------------------------------------------------------
    obs_xy = torch.zeros(n, M, 2, device=device)
    obs_r = torch.zeros(n, M, device=device)
    obs_on = torch.zeros(n, M, dtype=torch.bool, device=device)
    n_obs = torch.randint(0, M + 1, (n,), generator=gen, device=device)
    start_xy = pos[:, 0, :2] - torch.stack((torch.cos(yaw[:, 0]), torch.sin(yaw[:, 0])), -1) * rules.start_dist[1]
    keep_xy = torch.cat((pos[:, :, :2], torch.where(decoy_on[..., None], decoy_pos[:, :, :2], torch.full_like(pos[:, :, :2], 1e6))), 1)
    for m in range(M):
        placed = torch.zeros(n, dtype=torch.bool, device=device)
        want = m < n_obs
        for _ in range(4):
            seg = (torch.rand(n, generator=gen, device=device) * (count - 1).clamp_min(1)).long()
            seg = torch.minimum(seg, (count - 2).clamp_min(0))
            p0, p1 = pos[batch, seg, :2], pos[batch, seg + 1, :2]
            v = p1 - p0
            length = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            u = v / length
            nrm = torch.stack((-u[:, 1], u[:, 0]), -1)
            t = _u((n, 1), 0.25, 0.75, gen, device)
            sign = torch.where(torch.rand(n, 1, generator=gen, device=device) < 0.5, -1.0, 1.0)
            off_o = sign * _u((n, 1), rules.obs_lateral[0], rules.obs_lateral[1], gen, device)
            r = _u((n,), rules.obs_radius[0], rules.obs_radius[1], gen, device)
            c = p0 + t * v + off_o * nrm
            ok = (c[:, None, :] - keep_xy).norm(dim=-1).min(dim=-1).values >= rules.obs_gate_clear + r
            ok &= (c - start_xy).norm(dim=-1) >= rules.obs_gate_clear + r
            if m > 0:
                dprev = (c[:, None, :] - obs_xy[:, :m, :]).norm(dim=-1)
                dprev = torch.where(obs_on[:, :m], dprev, torch.full_like(dprev, 1e6))
                ok &= dprev.min(dim=-1).values >= rules.obs_spacing
            take = ok & want & ~placed & (count >= 2)
            obs_xy[take, m], obs_r[take, m], obs_on[take, m] = c[take], r[take], True
            placed |= take

    start_d = _u((n,), rules.start_dist[0], rules.start_dist[1], gen, device)
    return dict(
        gate_pos=pos, gate_yaw=yaw, n_gates=count, gate_size=gate_size,
        obs_xy=obs_xy, obs_r=obs_r, obs_on=obs_on, start_dist=start_d,
        marker_turn=turn, marker_climb=climb, marker_fork=fork_side,
        decoy_pos=decoy_pos, decoy_on=decoy_on, hall=hall,
    )


def generate_tracks(n: int, rules: TrackRules, capability: torch.Tensor, gen: torch.Generator, device) -> dict[str, torch.Tensor]:
    """Generate n tracks; tracks that end up with < min_gates are regenerated."""
    out = _generate_once(n, rules, capability, gen, device)
    for _ in range(10):
        bad = (out["n_gates"] < rules.min_gates).nonzero().squeeze(-1)
        if len(bad) == 0:
            break
        new = _generate_once(len(bad), rules, capability[bad], gen, device)
        for k in out:
            out[k][bad] = new[k]
    return out


def track_length(tr: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of gate-to-gate distances over the valid gates [N]."""
    p = tr["gate_pos"][:, :-1]
    seg = (p[:, 1:] - p[:, :-1]).norm(dim=-1)
    idx = torch.arange(seg.shape[1], device=p.device)[None]
    return (seg * (idx < (tr["n_gates"][:, None] - 1))).sum(-1)
