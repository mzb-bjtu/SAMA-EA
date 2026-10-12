"""Hierarchical policy network for legal search decisions."""
from __future__ import annotations

import torch
import torch.nn as nn

class MetaPolicyNet(nn.Module):

    def __init__(self, global_dim: int = 9, hidden: int = 32,
                 n_alg: int = 2, n_op: int = 18, n_op_par: int = 5,
                 n_dec: int = 4, n_dec_par: int = 5):
        super().__init__()
        self.n_alg = n_alg
        self.n_op = n_op
        self.n_dec = n_dec

        self.backbone = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())

        self.logit_alg = nn.Linear(hidden, n_alg)

        self.op_head = nn.Linear(hidden + n_alg, n_op)
        self.par_head = nn.Linear(hidden + n_alg, n_op_par)
        self.dec_head = nn.Linear(hidden + n_alg, n_dec)
        self.dec_par_head = nn.Linear(hidden + n_alg, n_dec_par)

    def forward(self, global_vec):
        h = self.backbone(global_vec)
        logits_alg = self.logit_alg(h)
        return h, logits_alg

    def conditioned(self, h, alg_idx: int):
        torch_mod = torch
        onehot = torch_mod.zeros(self.n_alg)
        onehot[alg_idx] = 1.0
        h_with_alg = torch_mod.cat([h, onehot.unsqueeze(0)], dim=-1)
        return (self.op_head(h_with_alg),
                self.par_head(h_with_alg),
                self.dec_head(h_with_alg),
                self.dec_par_head(h_with_alg))

def build_policy(device: str = "cpu", **kwargs) -> MetaPolicyNet:
    return MetaPolicyNet(**kwargs).to(device)
