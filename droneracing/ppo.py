"""Minimal PPO (own implementation) with asymmetric critic and running obs normalization."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOCfg:
    n_steps: int = 32
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    init_log_std: float = -0.7
    hidden: tuple[int, ...] = (256, 256, 128)
    critic_hidden: tuple[int, ...] | None = None  # training-only critic size (default = hidden)


class RunningNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var.copy_((self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-10, 10)


def mlp(i: int, o: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, policy_dim: int, critic_dim: int, act_dim: int, cfg: PPOCfg):
        super().__init__()
        self.actor = mlp(policy_dim, act_dim, tuple(cfg.hidden))
        self.critic = mlp(critic_dim, 1, tuple(cfg.critic_hidden or cfg.hidden))
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))
        self.pn, self.cn = RunningNorm(policy_dim), RunningNorm(critic_dim)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(self.pn(obs)), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, cobs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.cn(cobs)).squeeze(-1)


class PPO:
    def __init__(self, env, cfg: PPOCfg = PPOCfg()):
        self.env, self.cfg, self.dev = env, cfg, env.device
        self.ac = ActorCritic(env.policy_dim, env.critic_dim, 4, cfg).to(self.dev)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        self.obs, self.cobs = env.compute_obs()

    @torch.no_grad()
    def rollout(self):
        c, env = self.cfg, self.env
        T, N = c.n_steps, env.n
        buf = {k: [] for k in ("obs", "cobs", "act", "logp", "rew", "term", "done", "val", "nval")}
        stats = {"ep_return": [], "gates": [], "frac": [], "finished": [], "crashed": [], "len": [], "speed": [], "twr": []}
        for _ in range(T):
            self.ac.pn.update(self.obs)
            self.ac.cn.update(self.cobs)
            d = self.ac.dist(self.obs)
            act = d.sample()
            logp = d.log_prob(act).sum(-1)
            val = self.ac.value(self.cobs)
            obs2, cobs2, rew, term, done, info = env.step(act)
            # bootstrap value of the pre-reset state for time-outs
            nval = torch.where(info["timeout"], self.ac.value(info["final_critic_obs"]), torch.zeros_like(rew))
            for k, v in zip(buf, (self.obs, self.cobs, act, logp, rew, term, done, val, nval)):
                buf[k].append(v)
            if done.any():
                m = done
                stats["ep_return"].append(info["ep_return"][m])
                stats["gates"].append(info["gates_passed"][m].float())
                stats["frac"].append((info["gates_passed"][m].float() / info["n_gates"][m].float()))
                stats["finished"].append(info["finished"][m].float())
                stats["crashed"].append(info["crashed"][m].float())
                stats["len"].append(info["ep_len"][m].float())
                stats["twr"].append(info["twr"][m])
            stats["speed"].append(info["speed"])
            self.obs, self.cobs = obs2, cobs2
        buf = {k: torch.stack(v) for k, v in buf.items()}
        last_val = self.ac.value(self.cobs)
        adv = torch.zeros_like(buf["rew"])
        gae = torch.zeros(N, device=self.dev)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else buf["val"][t + 1]
            not_done = 1.0 - buf["done"][t].float()
            # at episode end: use bootstrap only for time-outs (nval), zero for terminations
            next_v = nv * not_done + buf["nval"][t]
            delta = buf["rew"][t] + c.gamma * next_v - buf["val"][t]
            gae = delta + c.gamma * c.lam * not_done * gae
            adv[t] = gae
        buf["adv"], buf["ret"] = adv, adv + buf["val"]
        return buf, stats

    def update(self, buf) -> dict[str, float]:
        c = self.cfg
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in buf.items() if k in ("obs", "cobs", "act", "logp", "adv", "ret", "val")}
        B = flat["obs"].shape[0]
        adv = flat["adv"]
        flat["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = B // c.minibatches
        kls, pl, vl = [], [], []
        for _ in range(c.epochs):
            perm = torch.randperm(B, device=self.dev)
            for i in range(c.minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                d = self.ac.dist(flat["obs"][idx])
                logp = d.log_prob(flat["act"][idx]).sum(-1)
                ratio = (logp - flat["logp"][idx]).exp()
                a = flat["adv"][idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                v = self.ac.value(flat["cobs"][idx])
                vf = (v - flat["ret"][idx]).pow(2).mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * d.entropy().sum(-1).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kls.append((flat["logp"][idx] - logp).mean().item())
                    pl.append(pg.item())
                    vl.append(vf.item())
            kl = sum(kls[-c.minibatches :]) / c.minibatches
            if c.desired_kl:
                self.lr = max(1e-5, self.lr / 1.5) if kl > 2 * c.desired_kl else (min(1e-3, self.lr * 1.5) if kl < 0.5 * c.desired_kl else self.lr)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr
        return {"kl": sum(kls) / len(kls), "pg": sum(pl) / len(pl), "vf": sum(vl) / len(vl), "lr": self.lr}

    def train(self, iterations: int, log_every: int = 10, ckpt=None, log_fn=print, on_iter=None):
        t0, n_env_steps = time.time(), 0
        for it in range(1, iterations + 1):
            if on_iter:
                on_iter(it, iterations, self)
            buf, st = self.rollout()
            u = self.update(buf)
            n_env_steps += self.cfg.n_steps * self.env.n
            if it % log_every == 0 or it == 1:
                cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(1, device=self.dev)  # noqa: E731
                row = {
                    "it": it, "steps": n_env_steps, "sps": n_env_steps / (time.time() - t0),
                    "ret": cat("ep_return").mean().item(), "gates": cat("gates").mean().item(),
                    "frac": cat("frac").mean().item(), "finish": cat("finished").mean().item(),
                    "crash": cat("crashed").mean().item(), "speed": torch.stack(st["speed"]).mean().item(),
                    "std": self.ac.log_std.exp().mean().item(), **u,
                }
                log_fn(row)
                if ckpt:
                    ckpt(it, self)
