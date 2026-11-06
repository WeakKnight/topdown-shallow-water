"""Procedural animation only; fluid interaction is evaluated by Slang on the GPU.

Positions are world XYZ, Y up. Support feet never inherit root translation.
Replace these poses with post-IK bone poses to integrate a real character.
"""
from dataclasses import dataclass, field
import math

import numpy as np


def water_margin(p):
    x, z = p
    shore = (1 - math.hypot(x / 10.2, z / 8.5)) * 8.5
    shore += 0.28 * math.sin(x * 1.1 + z * 0.7) + 0.18 * math.sin(z * 1.8)
    a = (math.hypot((x + 3.4) / 1.25, (z - 1.8) / 0.95) - 1) * 0.95
    b = (math.hypot((x - 4.0) / 0.75, (z + 2.8) / 1.15) - 1) * 0.75
    return min(shore, a, b)


def bottom_height(p, depth, open_water=False):
    t = 1.0 if open_water else min(1.0, max(0.0, water_margin(p) / 0.35))
    return -depth * t * t * (3 - 2 * t)


@dataclass
class Foot:
    position: np.ndarray
    previous: np.ndarray
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    half_size: tuple = (0.10, 0.22)
    half_height: float = 0.09
    swinging: bool = False
    phase: float = 0.0

    @property
    def immersion(self):
        return float(np.clip((self.half_height - self.position[1]) / (2 * self.half_height), 0, 1))


class Biped:
    def __init__(self, cadence=2.0, lift_height=0.28):
        self.cadence = cadence  # Total left + right steps per second at walking speed.
        self.lift_height = lift_height
        self.reset()

    def reset(self, root=(0.0, 0.0), depth=0.12, open_water=False):
        self.heading = np.array([0.0, -1.0])
        self.feet = []
        for side in (-1, 1):
            xz = np.array(root) + np.array([side * 0.19, 0.0])
            p = np.array([xz[0], bottom_height(xz, depth, open_water) + 0.09, xz[1]])
            self.feet.append(Foot(p, p.copy(), target=xz.copy()))
        self.active = None
        self.next_foot = 0
        self.cooldown = 0.0
        self.landings = [0, 0]

    def update(self, root, velocity, dt, depth=0.12, open_water=False):
        root, velocity = np.asarray(root), np.asarray(velocity)
        speed = float(np.linalg.norm(velocity))
        for foot in self.feet:
            foot.previous = foot.position.copy()
        if speed > 0.05:
            angle = math.atan2(self.heading[0], -self.heading[1])
            wanted = math.atan2(velocity[0], -velocity[1])
            delta = (wanted - angle + math.pi) % (2 * math.pi) - math.pi
            angle += float(np.clip(delta, -6 * dt, 6 * dt))
            self.heading = np.array([math.sin(angle), -math.cos(angle)])
        right = np.array([-self.heading[1], self.heading[0]])
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.active is None:
            # Finish both feet in a comfortable stance after releasing movement.
            i = self.next_foot
            desired = root + right * (-0.19 if i == 0 else 0.19)
            displaced = np.linalg.norm(self.feet[i].position[[0, 2]] - desired) > 0.13
            if speed <= 0.05 and not displaced:
                other = 1 - i
                other_desired = root + right * (-0.19 if other == 0 else 0.19)
                if np.linalg.norm(self.feet[other].position[[0, 2]] - other_desired) > 0.13:
                    i, desired, displaced = other, other_desired, True
            if self.cooldown <= 0 and (speed > 0.05 or displaced):
                frequency = self.cadence * np.clip(0.65 + 0.35 * speed / 1.4, 0.65, 2.0)
                # Preserve a plausible stride even at low cadence + sprint speed.
                frequency = max(frequency, speed / 0.85)
                interval = 1.0 / frequency
                self.duration = interval * 0.78
                self.cooldown = interval
                self.elapsed = 0.0
                self.active = i
                self.start = self.feet[i].position.copy()
                self.start_forward = self.feet[i].forward.copy()
                self.end_forward = self.heading.copy()
                self.end_xz = desired + velocity * interval * 0.65
                self.end_y = bottom_height(self.end_xz, depth, open_water) + self.feet[i].half_height
                self.feet[i].target = self.end_xz.copy()
                self.feet[i].swinging = True
        if self.active is not None:
            i = self.active
            foot = self.feet[i]
            self.elapsed += dt
            t = min(1.0, self.elapsed / self.duration)
            blend = t * t * (3 - 2 * t)
            foot.position[[0, 2]] = self.start[[0, 2]] * (1 - blend) + self.end_xz * blend
            # Lift early, translate mostly above water, then lower into the water.
            foot.position[1] = self.start[1] * (1 - blend) + self.end_y * blend + self.lift_height * math.sin(math.pi * t)
            a = math.atan2(self.start_forward[0], -self.start_forward[1])
            b = math.atan2(self.end_forward[0], -self.end_forward[1])
            angle = a + ((b - a + math.pi) % (2 * math.pi) - math.pi) * blend
            foot.forward = np.array([math.sin(angle), -math.cos(angle)])
            foot.phase = t
            if t >= 1:
                foot.position[1] = self.end_y
                foot.swinging = False
                self.landings[i] += 1
                self.active = None
                self.next_foot = 1 - i
        else:
            # Changing depth is a scene edit. Re-seat planted feet without impulses.
            for foot in self.feet:
                foot.position[1] = bottom_height(foot.position[[0, 2]], depth, open_water) + foot.half_height
                foot.previous[1] = foot.position[1]
        return self.feet
