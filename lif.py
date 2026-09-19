"""lif.py -- current-based leaky integrate-and-fire dynamics on a signed sparse W.

    tau_m dv/dt = (v_rest - v) + g
    tau_s dg/dt = -g + tau_m * sum_j W_ij s_j(t) + tau_m * w_ext * xi_i(t)

Exact-exponential synaptic decay, forward-Euler membrane, absolute refractory
period. Parameters are borrowed from insect LIF whole-brain models and are
placeholders for the cervid case.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LIFNetwork:
    def __init__(self, W: sp.csr_matrix, dt: float = 0.5, tau_m: float = 20.0,
                 tau_syn: float = 5.0, v_rest: float = -52.0, v_reset: float = -52.0,
                 v_th: float = -45.0, t_ref: float = 2.2, w_ext: float = 0.275,
                 seed: int = 0):
        self.W = W.tocsr()
        self.n = W.shape[0]
        self.dt, self.tau_m, self.tau_syn = dt, tau_m, tau_syn
        self.v_rest, self.v_reset, self.v_th = v_rest, v_reset, v_th
        self.t_ref, self.w_ext = t_ref, w_ext
        self.k = tau_m / tau_syn      # jump scaling so peak PSP ~ w [mV]
        self.decay = np.exp(-dt / tau_syn)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.v = np.full(self.n, self.v_rest)
        self.g = np.zeros(self.n)
        self.ref = np.zeros(self.n)
        self.spk = np.zeros(self.n, dtype=bool)

    def step(self, ext_rate_hz: np.ndarray) -> np.ndarray:
        """Advance one dt. ext_rate_hz: (N,) Poisson afferent rate per neuron."""
        xi = self.rng.random(self.n) < ext_rate_hz * self.dt * 1e-3
        self.g += self.k * (self.W @ self.spk.astype(float) + self.w_ext * xi)
        dv = (self.v_rest - self.v + self.g) / self.tau_m * self.dt
        self.v = np.where(self.ref > 0, self.v_reset, self.v + dv)
        self.g *= self.decay
        self.ref = np.maximum(self.ref - self.dt, 0.0)
        self.spk = (self.v >= self.v_th) & (self.ref <= 0)
        self.v[self.spk] = self.v_reset
        self.ref[self.spk] = self.t_ref
        return self.spk
