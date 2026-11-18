"""Corrupt-frame check (radio-link interference: a band of rows replaced by noise or by a smeared copy of one row).

A frame is rejected when a band of at least ``min_rows`` consecutive rows is (a) noise-like: the horizontal gradient of the row is far above the
frame median, or (b) smeared: the row is (nearly) identical to the row above. The filter then skips the detections of the frame (MonoRace rejects
outliers; a wrong corner under the band cannot be labelled, so training cannot fix this).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def corrupt_frames(img: torch.Tensor, min_frac: float = 0.06, noise_k: float = 3.0, smear_eps: float = 0.004) -> torch.Tensor:
    """img [n,3,H,W] in 0..1 -> bool [n] (True = corrupt)."""
    g = img.mean(1)  # [n,H,W]
    n, H, W = g.shape
    gx = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(-1)  # [n,H] horizontal activity per row
    gy = (g[:, 1:] - g[:, :-1]).abs().mean(-1)  # [n,H-1] difference to the row above
    med = gx.median(dim=1, keepdim=True).values.clamp_min(0.004)
    noisy = gx > noise_k * med + 0.05  # uniform noise rows have ~0.33
    smear = gy < smear_eps
    # a smeared band also has zero horizontal change only if the source row is flat; the vertical identity is the signature
    k = max(3, int(min_frac * H))

    def run(m):  # longest run of True per sample
        best = torch.zeros(n, dtype=torch.long, device=img.device)
        cur = torch.zeros(n, dtype=torch.long, device=img.device)
        for i in range(m.shape[1]):
            cur = torch.where(m[:, i], cur + 1, torch.zeros_like(cur))
            best = torch.maximum(best, cur)
        return best

    return (run(noisy) >= k) | (run(smear) >= k)


@torch.no_grad()
def corrupt_rows(img: torch.Tensor, min_frac: float = 0.06, noise_k: float = 3.0, smear_eps: float = 0.004, pad: int = 6) -> torch.Tensor:
    """img [n,3,H,W] -> bool [n,H]: rows inside a corrupt band (run of >= min_frac*H noisy or smeared rows), widened by ``pad`` rows.
    Detections in these rows are dropped; the rest of the frame stays usable (MonoRace: the network finds no gate in the band)."""
    g = img.mean(1)
    n, H, W = g.shape
    gx = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(-1)
    gy = torch.cat(((g[:, 1:] - g[:, :-1]).abs().mean(-1), torch.ones(n, 1, device=img.device)), 1)
    med = gx.median(dim=1, keepdim=True).values.clamp_min(0.004)
    bad = (gx > noise_k * med + 0.05) | (gy < smear_eps)
    k = max(3, int(min_frac * H))
    run = torch.zeros(n, H, dtype=torch.long, device=img.device)
    cur = torch.zeros(n, dtype=torch.long, device=img.device)
    for i in range(H):  # forward run length
        cur = torch.where(bad[:, i], cur + 1, torch.zeros_like(cur))
        run[:, i] = cur
    keep = torch.zeros(n, H, dtype=torch.bool, device=img.device)
    rem = torch.zeros(n, dtype=torch.long, device=img.device)
    for i in range(H - 1, -1, -1):  # mark every row of a run that reaches length k
        start = run[:, i] >= k
        rem = torch.where(start & (rem == 0), run[:, i], rem)
        keep[:, i] = rem > 0
        rem = (rem - 1).clamp_min(0)
    if pad > 0:
        keep = torch.nn.functional.max_pool1d(keep.float()[:, None], 2 * pad + 1, 1, pad)[:, 0] > 0
    return keep
