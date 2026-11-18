"""Recurrent PPO on pixels: RGB + IMU + previous action -> body-rate/thrust command (no teacher).

Actor = CNN + MLP + GRU (same network as the DAgger student); critic = MLP on the privileged state
(asymmetric, training only). Sequences of ``n_steps`` per env are replayed with truncated BPTT using the
stored GRU state at the start of the rollout. Optional auxiliary supervised heads (marker token and
next-gate position, labels from the simulator) shape the visual features; they never touch the policy
input at run time.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import augment
from .ppo import RunningNorm, mlp
from .quad import quat_to_rot
from .student import StudentNet


class PixelAC(nn.Module):
    def __init__(self, critic_dim: int, img_hw, act_dim: int = 4, init_log_std: float = -0.7):
        super().__init__()
        self.net = StudentNet(act_dim, 128, img_hw)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.critic = mlp(critic_dim, 1, (256, 256, 128))
        self.cn = RunningNorm(critic_dim)

    def feats(self, img, prop):
        """img [B,T,3,H,W] in [0,1], prop [B,T,10] -> x [B,T,160]"""
        B, T = img.shape[:2]
        f = self.net.img_fc(self.net.enc(img.reshape(B * T, *img.shape[2:]) - 0.5)).reshape(B, T, -1)
        return torch.cat((f, self.net.prop_fc(prop)), -1)

    def run(self, x, h, reset=None):
        """x [B,T,F]; h [1,B,H]; reset [B,T] bool: zero the state *before* step t. -> y [B,T,H], h."""
        if reset is None:
            return self.net.gru(x, h)
        ys = []
        for t in range(x.shape[1]):
            h = h * (~reset[:, t]).float()[None, :, None]
            y, h = self.net.gru(x[:, t : t + 1], h)
            ys.append(y)
        return torch.cat(ys, 1), h

    def heads(self, y):
        mean = torch.tanh(self.net.head(y))
        return mean, {"token": self.net.aux_token(y), "gate": self.net.aux_gate(y)}

    def value(self, cobs):
        return self.critic(self.cn(cobs)).squeeze(-1)


def aux_loss(aux, tok, gat):
    tl = aux["token"]
    l_tok = (
        F.cross_entropy(tl[..., 0:4].reshape(-1, 4), tok[..., 0:4].argmax(-1).reshape(-1))
        + F.cross_entropy(tl[..., 4:7].reshape(-1, 3), tok[..., 4:7].argmax(-1).reshape(-1))
        + F.cross_entropy(tl[..., 7:10].reshape(-1, 3), tok[..., 7:10].argmax(-1).reshape(-1))
        + F.binary_cross_entropy_with_logits(tl[..., 10], tok[..., 10])
    )
    return l_tok, F.smooth_l1_loss(aux["gate"], gat)


def run_pixel_ppo(env, render_fn, img_hw, out: Path, iters: int, T: int = 32, lr: float = 3e-4, aug: float = 1.0,
                  aux_coef: float = 0.5, epochs: int = 4, minibatches: int = 4, gamma: float = 0.99, lam: float = 0.95,
                  clip: float = 0.2, desired_kl: float = 0.01, log_every: int = 5, post_step=None, on_iter=None):
    dev, N = env.device, env.n
    ac = PixelAC(env.critic_dim, img_hw).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    cur_lr = lr
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "log.csv", "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["it", "steps", "sps", "ret", "gates", "frac", "finish", "crash", "speed", "std", "kl", "tok", "gate", "lr"])
    _, cobs = env.compute_obs()
    h = torch.zeros(1, N, ac.net.hidden, device=dev)
    t0, steps = time.time(), 0
    for it in range(1, iters + 1):
        if on_iter:
            on_iter(it, iters)
        h0 = h.clone()
        B = {k: [] for k in ("img", "prop", "act", "logp", "rew", "done", "val", "nval", "cobs", "tok", "gat")}
        st = {k: [] for k in ("ret", "gates", "frac", "fin", "crash")}
        speed = []
        with torch.no_grad():
            for _ in range(T):
                raw = render_fn()
                img = augment(raw, env.quad.omega, aug)
                prop = torch.cat((env.imu_obs(), env.prev_action), -1)
                ac.cn.update(cobs)
                x = ac.feats(img[:, None], prop[:, None])
                y, h = ac.run(x, h)
                mean, _ = ac.heads(y)
                dist = torch.distributions.Normal(mean[:, 0], ac.log_std.exp().expand(N, -1))
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = ac.value(cobs)
                idx = torch.minimum(env.gate_idx, env._last_slot())
                c, _, _ = env._gate_frame(idx)
                Rt = quat_to_rot(env.quad.quat).transpose(1, 2)
                rel = ((Rt @ (c - env.quad.pos).unsqueeze(-1)).squeeze(-1) / 5.0).clamp(-4.0, 4.0)
                tok = env._tokens(idx)
                _, cobs2, rew, term, done, info = env.step(act)
                if post_step is not None:
                    post_step(env)
                nval = torch.where(info["timeout"], ac.value(info["final_critic_obs"]), torch.zeros_like(rew))
                for k, v in zip(B, ((img * 255).to(torch.uint8), prop, act, logp, rew, done, val, nval, cobs, tok, rel)):
                    B[k].append(v)
                h = h * (~done).float()[None, :, None]
                if done.any():
                    st["ret"].append(info["ep_return"][done]); st["gates"].append(info["gates_passed"][done].float())
                    st["frac"].append(info["gates_passed"][done].float() / info["n_gates"][done].float())
                    st["fin"].append(info["finished"][done].float()); st["crash"].append(info["crashed"][done].float())
                speed.append(info["speed"])
                cobs = cobs2
            last_val = ac.value(cobs)
        S = {k: torch.stack(v, 1) for k, v in B.items()}  # [N,T,...]
        rew, done, val, nval = S["rew"].T, S["done"].T, S["val"].T, S["nval"].T
        adv = torch.zeros_like(rew)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            nd = 1.0 - done[t].float()
            delta = rew[t] + gamma * (nv * nd + nval[t]) - val[t]
            gae = delta + gamma * lam * nd * gae
            adv[t] = gae
        ret = (adv + val).T
        adv = adv.T
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        reset = torch.zeros(N, T, dtype=torch.bool, device=dev)
        reset[:, 1:] = S["done"][:, :-1]
        kls, ltok, lgate = [], [], []
        mb = N // minibatches
        for _ in range(epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(minibatches):
                j = perm[i * mb : (i + 1) * mb]
                img = S["img"][j].float() / 255.0  # already augmented at collection time
                x = ac.feats(img, S["prop"][j])
                y, _ = ac.run(x, h0[:, j].contiguous(), reset[j])
                mean, aux = ac.heads(y)
                dist = torch.distributions.Normal(mean, ac.log_std.exp().expand_as(mean))
                logp = dist.log_prob(S["act"][j]).sum(-1)
                ratio = (logp - S["logp"][j]).exp()
                a = adv[j]
                pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                vf = (ac.value(S["cobs"][j]) - ret[j]).pow(2).mean()
                l_tok, l_gate = aux_loss(aux, S["tok"][j], S["gat"][j])
                loss = pg + vf + aux_coef * (l_tok + l_gate)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    kls.append((S["logp"][j] - logp).mean().item())
                ltok.append(l_tok.item()); lgate.append(l_gate.item())
            kl = sum(kls[-minibatches:]) / minibatches
            if desired_kl:
                cur_lr = max(1e-5, cur_lr / 1.5) if kl > 2 * desired_kl else (min(1e-3, cur_lr * 1.5) if kl < 0.5 * desired_kl else cur_lr)
                for g in opt.param_groups:
                    g["lr"] = cur_lr
        steps += T * N
        if it % log_every == 0 or it == 1:
            cat = lambda k: torch.cat(st[k]).mean().item() if st[k] else float("nan")  # noqa: E731
            row = [it, steps, steps / (time.time() - t0), cat("ret"), cat("gates"), cat("frac"), cat("fin"), cat("crash"),
                   torch.stack(speed).mean().item(), ac.log_std.exp().mean().item(), sum(kls) / len(kls),
                   sum(ltok) / len(ltok), sum(lgate) / len(lgate), cur_lr]
            wr.writerow(row); f.flush()
            print(f"it {it:4d} | {row[2]:6.0f} sps | ret {row[3]:7.1f} | gates {row[4]:4.1f} ({row[5]*100:3.0f}%) | "
                  f"finish {row[6]*100:3.0f}% | crash {row[7]*100:3.0f}% | v {row[8]:4.1f} | std {row[9]:.2f} | "
                  f"tok {row[11]:.2f} gate {row[12]:.3f}", flush=True)
            torch.save({"ac": ac.state_dict(), "it": it}, out / "pixel_ac.pt")
    return ac
