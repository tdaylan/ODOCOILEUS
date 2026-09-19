"""run.py -- closed-loop embodiment: env -> Poisson afferents -> LIF connectome -> motor pools -> env."""
from __future__ import annotations

import argparse

import numpy as np

from connectome import build
from env import N_VIS, ForestEnv
from lif import LIFNetwork

BG_HZ = 1000.0     # tonic afferent drive to all neurons (placeholder; near-critical, see README)
GAIN_OLF = 400.0   # Hz per unit plume concentration into OB
GAIN_VIS = 1500.0  # Hz per unit looming into LGN
TAU_RATE = 100.0   # ms, motor readout filter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C = build(n=args.n, seed=args.seed)
    net = LIFNetwork(C.W, dt=0.5, seed=args.seed)
    env = ForestEnv(seed=args.seed)
    print(C.describe())

    sl, hemi = C.slices, C.hemi
    ob, lgn = np.arange(C.n)[sl["OB"]], np.arange(C.n)[sl["LGN"]]
    lgn_bin = np.array([(i // 2) % (N_VIS // 2) + (N_VIS // 2) * hemi[j] for i, j in enumerate(lgn)])
    m1, mlr, cpg = (np.arange(C.n)[sl[k]] for k in ("M1", "MLR", "CPG"))
    m1_l, m1_r = m1[hemi[m1] == 0], m1[hemi[m1] == 1]

    steps_per_act = int(20.0 / net.dt)          # 20 ms actuation period
    alpha = 20.0 / TAU_RATE
    r_l = r_r = r_cpg = r_mlr = 0.0
    counts = np.zeros(C.n)
    n_env = int(args.seconds / 0.02)

    for _ in range(n_env):
        s = env.sense()
        ext = np.full(C.n, BG_HZ)
        ext[ob] += GAIN_OLF * s["olf"][hemi[ob]]
        ext[lgn] += GAIN_VIS * s["vis"][lgn_bin]
        window = np.zeros(C.n)
        for _ in range(steps_per_act):
            window += net.step(ext)
        counts += window
        hz = window / 0.02
        r_l += alpha * (hz[m1_l].mean() - r_l)
        r_r += alpha * (hz[m1_r].mean() - r_r)
        r_cpg += alpha * (hz[cpg].mean() - r_cpg)
        r_mlr += alpha * (hz[mlr].mean() - r_mlr)
        turn = float(np.tanh((r_l - r_r) / 5.0))
        fwd = float(np.clip(r_cpg / 10.0, 0.0, 1.0))
        flight = r_mlr > 15.0
        info = env.step(fwd, turn, flight)
        if info["dead"]:
            break

    print(f"t={env.t:.1f}s  food={env.eaten}  caught={env.dead}")
    hz_reg = {k: counts[v].mean() / env.t for k, v in sl.items()}
    print("mean rates [Hz]:", {k: round(v, 1) for k, v in hz_reg.items()})


if __name__ == "__main__":
    main()
