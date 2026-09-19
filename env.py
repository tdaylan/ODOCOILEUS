"""env.py -- minimal 2D closed-loop world: forage on scent plumes, evade a stalking predator.

Conventions: heading is CCW from +x; positive bearing / positive turn = LEFT.
Sensors: bilateral olfaction (2 nostrils) + 12-bin panoramic looming (+-150 deg).
Actuators: fwd in [0,1], turn in [-1,1], flight in {0,1}.
"""
from __future__ import annotations

import numpy as np

N_VIS = 12
FOV = np.deg2rad(150.0)


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class ForestEnv:
    def __init__(self, arena: float = 200.0, n_food: int = 3, seed: int = 0):
        self.arena, self.n_food = arena, n_food
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        a = self.arena
        self.pos = np.array([a / 2, a / 2])
        self.heading = self.rng.uniform(-np.pi, np.pi)
        self.food = self.rng.uniform(0.1 * a, 0.9 * a, size=(self.n_food, 2))
        self.wolf = self.rng.uniform(0, a, size=2)
        self.t, self.eaten, self.dead = 0.0, 0, False

    def _plume(self, p: np.ndarray, sigma: float = 25.0) -> float:
        d2 = ((self.food - p) ** 2).sum(1)
        return float(np.exp(-d2 / (2 * sigma**2)).sum())

    def sense(self) -> dict:
        h = self.heading
        left = self.pos + 0.3 * np.array([np.cos(h + np.pi / 2), np.sin(h + np.pi / 2)])
        right = self.pos + 0.3 * np.array([np.cos(h - np.pi / 2), np.sin(h - np.pi / 2)])
        olf = np.array([self._plume(left), self._plume(right)])  # [L, R]
        vis = np.zeros(N_VIS)
        rel = self.wolf - self.pos
        bearing = wrap(np.arctan2(rel[1], rel[0]) - h)
        if abs(bearing) < FOV:
            b = int((FOV - bearing) / (2 * FOV) * N_VIS)  # bins 0-5 left, 6-11 right
            vis[min(b, N_VIS - 1)] = np.exp(-np.linalg.norm(rel) / 40.0)
        return {"olf": olf, "vis": vis}

    def step(self, fwd: float, turn: float, flight: bool, dt: float = 0.02) -> dict:
        self.heading = wrap(self.heading + 2.0 * turn * dt)
        speed = 2.5 if flight else 1.0 * fwd
        self.pos = np.clip(self.pos + speed * dt * np.array([np.cos(self.heading), np.sin(self.heading)]),
                           0, self.arena)
        rel = self.pos - self.wolf
        d = np.linalg.norm(rel) + 1e-9
        self.wolf = self.wolf + (1.4 if d < 30 else 0.6) * dt * rel / d
        reward = 0.0
        for i, f in enumerate(self.food):
            if np.linalg.norm(f - self.pos) < 2.0:
                self.food[i] = self.rng.uniform(0.1 * self.arena, 0.9 * self.arena, 2)
                self.eaten += 1
                reward += 1.0
        self.dead = d < 1.0
        self.t += dt
        return {"reward": reward, "dead": self.dead, "wolf_dist": d}
