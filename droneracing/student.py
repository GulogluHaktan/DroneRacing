"""RGB + IMU student policy (recurrent) distilled from the privileged teacher.

Inputs per step: image [3,H,W], IMU [6], previous action [4]. A GRU carries memory (markers seen
earlier, velocity estimate). Auxiliary heads (trained from privileged labels) shape the features:
marker token of the current gate (turn/climb/fork/finish) and the next gate's body-frame position.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StudentNet(nn.Module):
    def __init__(self, act_dim: int = 4, hidden: int = 128, img_hw=(48, 64)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            feat = self.enc(torch.zeros(1, 3, *img_hw)).shape[-1]
        self.img_fc = nn.Sequential(nn.Linear(feat, 128), nn.ReLU())
        self.prop_fc = nn.Sequential(nn.Linear(6 + act_dim, 32), nn.ReLU())
        self.gru = nn.GRU(128 + 32, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, act_dim))
        self.aux_token = nn.Linear(hidden, 4 + 3 + 3 + 1)  # turn, climb, fork, finish
        self.aux_gate = nn.Linear(hidden, 3)  # next gate position in the body frame (/5 m)
        self.hidden = hidden

    def forward(self, img, prop, h=None):
        """img [B,T,3,H,W], prop [B,T,10], h [1,B,hidden] -> action [B,T,4], aux, h."""
        B, T = img.shape[:2]
        f = self.img_fc(self.enc(img.reshape(B * T, *img.shape[2:]))).reshape(B, T, -1)
        x = torch.cat((f, self.prop_fc(prop)), -1)
        y, h = self.gru(x, h)
        return torch.tanh(self.head(y)), {"token": self.aux_token(y), "gate": self.aux_gate(y)}, h
