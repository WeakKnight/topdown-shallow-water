"""Seeded wandering and procedural poses; all water interaction stays on the GPU."""
import numpy as np

from gait import Biped, water_margin


class Wanderer:
    def __init__(self, index, extent=24.0, depth=0.12, open_water=False):
        self.index, self.extent = index, extent
        self.gait = Biped(cadence=1.8 + index * 0.2, lift_height=0.24 + index * 0.025)
        self.reset(depth, open_water)

    def reset(self, depth, open_water):
        self.rng = np.random.default_rng(731 + self.index * 97)
        self.position = np.array(((-2.0, -1.5), (2.0, -1.5), (0.0, 2.5))[self.index])
        self.velocity = np.zeros(2)
        self.target = None
        self.pause = self.index * 0.21
        self.gait.reset(self.position, depth, open_water)

    def safe(self, point, open_water):
        return (np.max(np.abs(point)) < self.extent * 0.5 - 1.0
                and (open_water or water_margin(point) > 0.9))

    def choose_target(self, open_water):
        span = np.minimum([5.2, 3.6], self.extent * 0.5 - 1.3)
        for _ in range(100):
            candidate = self.rng.uniform(-span, span)
            if np.linalg.norm(candidate - self.position) < 1.5:
                continue
            # Check the entire route, including the islands, with foot clearance.
            if all(self.safe(p, open_water) for p in np.linspace(self.position, candidate, 48)):
                self.target = candidate
                self.speed = self.rng.uniform(0.85, 1.45)
                return
        self.target = None

    def update(self, dt, depth, open_water):
        previous = self.position.copy()
        self.pause = max(0.0, self.pause - dt)
        if self.target is None and self.pause <= 0:
            self.choose_target(open_water)
        desired = np.zeros(2)
        if self.target is not None:
            delta = self.target - self.position
            distance = np.linalg.norm(delta)
            if distance < 0.15:
                self.target = None
                self.pause = self.rng.uniform(0.1, 0.65)
            else:
                desired = delta / distance * min(self.speed, distance * 2.5)
        change = desired - self.velocity
        self.velocity += change * min(1.0, 3.0 * dt / max(np.linalg.norm(change), 1e-8))
        candidate = self.position + self.velocity * dt
        if self.safe(candidate, open_water):
            self.position = candidate
        else:
            self.velocity[:] = 0
            self.target = None
        return self.gait.update(self.position, (self.position - previous) / dt, dt, depth, open_water)
