"""connectome.py -- synthetic mesoscale-prior connectome for O. virginianus.

No cervid synaptic-resolution connectome exists. This module samples a
degree-corrected stochastic block model (DC-SBM) over a region-level
projection graph, with Dale's-law sign assignment and a hemispheric
(ipsi/contra) bias per projection. Output: signed sparse W[post, pre] in mV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# Fraction of total neurons per mesoscale compartment (placeholder priors).
REGIONS = {
    "OB": 0.06, "LGN": 0.03, "V1": 0.14, "SC": 0.05, "PIR": 0.06, "AMY": 0.05,
    "HPC": 0.07, "PFC": 0.13, "STR": 0.08, "M1": 0.12, "MLR": 0.06, "CPG": 0.15,
}

# (src, dst, block density p, contralateral fraction in [0, 1])
PROJECTIONS = [
    ("OB", "PIR", 0.05, 0.15), ("PIR", "STR", 0.04, 0.20), ("PIR", "AMY", 0.03, 0.50),
    ("PIR", "HPC", 0.03, 0.50), ("STR", "M1", 0.04, 0.15), ("STR", "MLR", 0.04, 0.30),
    ("LGN", "V1", 0.05, 0.10), ("LGN", "SC", 0.05, 0.85), ("V1", "PFC", 0.03, 0.50),
    ("V1", "AMY", 0.03, 0.50), ("SC", "M1", 0.05, 0.85), ("SC", "AMY", 0.04, 0.60),
    ("AMY", "MLR", 0.05, 0.50), ("AMY", "PFC", 0.03, 0.50), ("HPC", "PFC", 0.03, 0.50),
    ("PFC", "STR", 0.03, 0.50), ("PFC", "M1", 0.03, 0.50), ("M1", "CPG", 0.05, 0.15),
    ("MLR", "CPG", 0.06, 0.50),
]
LOCAL_P = 0.03  # intra-region recurrent density


@dataclass
class Connectome:
    W: sp.csr_matrix          # (N, N), W[i, j] = signed PSP amplitude j -> i [mV]
    slices: dict              # region name -> slice into neuron index space
    hemi: np.ndarray          # (N,) in {0 = left, 1 = right}
    sign: np.ndarray          # (N,) in {+1, -1}, Dale's law

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def describe(self) -> str:
        nnz = self.W.nnz
        return (f"N={self.n}  edges={nnz}  <k_out>={nnz / self.n:.1f}  "
                f"frac_inh={(self.sign < 0).mean():.2f}")


def build(n: int = 4000, seed: int = 0, w_syn: float = 0.275,
          frac_inh: float = 0.2, sigma_theta: float = 0.5) -> Connectome:
    rng = np.random.default_rng(seed)
    names = list(REGIONS)
    frac = np.array([REGIONS[k] for k in names])
    counts = np.floor(frac / frac.sum() * n).astype(int)
    counts[-1] += n - counts.sum()
    bounds = np.concatenate([[0], np.cumsum(counts)])
    slices = {k: slice(int(bounds[i]), int(bounds[i + 1])) for i, k in enumerate(names)}

    hemi = np.zeros(n, dtype=int)
    for s in slices.values():
        hemi[s] = rng.permutation(np.arange(s.stop - s.start) % 2)
    sign = np.where(rng.random(n) < frac_inh, -1.0, 1.0)
    theta_out = rng.lognormal(0.0, sigma_theta, n)  # degree-correction propensities
    theta_in = rng.lognormal(0.0, sigma_theta, n)

    rows, cols = [], []
    for src, dst, p, contra in PROJECTIONS + [(r, r, LOCAL_P, 0.5) for r in names]:
        ss, ds = slices[src], slices[dst]
        ns, nd = ss.stop - ss.start, ds.stop - ds.start
        k = 2 * rng.binomial(ns * nd, p)  # oversample 2x, thin by hemispheric factor
        if k == 0:
            continue
        pre = rng.choice(np.arange(ss.start, ss.stop), size=k, p=theta_out[ss] / theta_out[ss].sum())
        post = rng.choice(np.arange(ds.start, ds.stop), size=k, p=theta_in[ds] / theta_in[ds].sum())
        f = np.where(hemi[pre] == hemi[post], 2 * (1 - contra), 2 * contra)
        keep = (rng.random(k) < f / 2) & (pre != post)
        rows.append(post[keep])
        cols.append(pre[keep])

    r, c = np.concatenate(rows), np.concatenate(cols)
    counts_mat = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()  # multi-synapse sums
    W = counts_mat @ sp.diags(sign * w_syn)
    return Connectome(W=W.tocsr(), slices=slices, hemi=hemi, sign=sign)


if __name__ == "__main__":
    print(build().describe())
