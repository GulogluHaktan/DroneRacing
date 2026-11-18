"""DAgger distillation loop: privileged teacher -> RGB + IMU student.

``render_fn() -> [N,3,H,W] float in [0,1]`` is the only camera dependency: Isaac TiledCamera in real
training (isaac/train_student_isaac.py), the analytic ray-caster only in CPU unit/smoke tests.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from .camera import augment
from .quad import quat_to_rot


def _losses(student, img, prop, tgt, tok, gat, burn: int):
    pred, aux, _ = student(img, prop, None)
    sl = slice(burn, None)  # first steps only warm up the GRU state
    l_bc = F.mse_loss(pred[:, sl], tgt[:, sl])
    tl, tk = aux["token"][:, sl], tok[:, sl]
    l_tok = (
        F.cross_entropy(tl[..., 0:4].reshape(-1, 4), tk[..., 0:4].argmax(-1).reshape(-1))
        + F.cross_entropy(tl[..., 4:7].reshape(-1, 3), tk[..., 4:7].argmax(-1).reshape(-1))
        + F.cross_entropy(tl[..., 7:10].reshape(-1, 3), tk[..., 7:10].argmax(-1).reshape(-1))
        + F.binary_cross_entropy_with_logits(tl[..., 10], tk[..., 10])
    )
    l_gate = F.smooth_l1_loss(aux["gate"][:, sl], gat[:, sl])
    return l_bc, l_tok, l_gate


def run_dagger(env, render_fn, teacher, student, out: Path, iters: int, T: int = 32, lr: float = 5e-4,
               beta0: float = 1.0, beta_iters: int = 50, aug: float = 1.0, log_every: int = 5, post_step=None,
               buffer_iters: int = 24, updates: int = 16, batch: int = 64, burn: int = 8):
    """DAgger with an aggregated replay buffer (last ``buffer_iters`` rollouts, raw uint8 frames) and
    ``updates`` minibatch gradient steps per iteration; frames are augmented again when sampled."""
    dev = env.device
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "log.csv", "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["it", "beta", "bc", "token", "gate", "crash", "finish"])
    n = env.n
    h = torch.zeros(1, n, student.hidden, device=dev)
    buf = []  # list of dicts of [n,T,...] tensors
    for it in range(1, iters + 1):
        beta = max(0.0, beta0 * (1 - (it - 1) / max(1, beta_iters)))
        imgs, props, tgt, tok, gat = [], [], [], [], []
        crash = fin = 0.0
        cnt = 0
        with torch.no_grad():
            for _ in range(T):
                raw = render_fn()
                img = augment(raw, env.quad.omega, aug)
                prop = torch.cat((env.imu_obs(), env.prev_action), -1)
                t_act = teacher.actor(teacher.pn(env.teacher_obs())).clamp(-1, 1)
                s_act, _, h = student(img[:, None], prop[:, None], h)
                use_t = (torch.rand(n, device=dev) < beta)[:, None]
                act = torch.where(use_t, t_act, s_act[:, 0])
                idx = torch.minimum(env.gate_idx, env._last_slot())
                c, _, _ = env._gate_frame(idx)
                Rt = quat_to_rot(env.quad.quat).transpose(1, 2)
                rel = ((Rt @ (c - env.quad.pos).unsqueeze(-1)).squeeze(-1) / 5.0).clamp(-4.0, 4.0)
                imgs.append((raw * 255).to(torch.uint8)); props.append(prop); tgt.append(t_act)
                tok.append(env._tokens(idx)); gat.append(rel)
                _, _, _, _, done, info = env.step(act)
                if post_step is not None:
                    post_step(env)
                h = h * (~done).float()[None, :, None]
                crash += info["crashed"].float().sum().item()
                fin += info["finished"].float().sum().item()
                cnt += int(done.sum().item())
        S = lambda xs: torch.stack(xs, 1)  # noqa: E731
        buf.append(dict(img=S(imgs), prop=S(props), tgt=S(tgt), tok=S(tok), gat=S(gat)))
        buf = buf[-buffer_iters:]
        omega_zero = torch.zeros(batch, 3, device=dev)
        stats = torch.zeros(3)
        for _ in range(updates):
            bi = torch.randint(0, len(buf), (batch,))
            ei = torch.randint(0, n, (batch,), device=dev)
            pick = lambda k: torch.stack([buf[int(b)][k][e] for b, e in zip(bi.tolist(), ei.tolist())])  # noqa: E731
            im = pick("img").float() / 255.0  # [B,T,3,H,W]
            B_, T_ = im.shape[:2]
            im = augment(im.reshape(B_ * T_, *im.shape[2:]), torch.zeros(B_ * T_, 3, device=dev), aug).reshape(im.shape)
            l_bc, l_tok, l_gate = _losses(student, im, pick("prop"), pick("tgt"), pick("tok"), pick("gat"), burn)
            loss = l_bc + 0.2 * l_tok + 0.5 * l_gate
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            stats += torch.tensor([l_bc.item(), l_tok.item(), l_gate.item()]) / updates
        c_ = max(cnt, 1)
        wr.writerow([it, round(beta, 3), *stats.tolist(), crash / c_, fin / c_])
        f.flush()
        if it % log_every == 0 or it == 1:
            print(f"it {it:4d} | beta {beta:.2f} | bc {stats[0]:.4f} | tok {stats[1]:.3f} | gate {stats[2]:.4f} "
                  f"| crash {crash/c_*100:3.0f}% | fin {fin/c_*100:3.0f}%", flush=True)
        torch.save({"student": student.state_dict(), "it": it}, out / "student.pt")
