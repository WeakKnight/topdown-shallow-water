"""GPU height-field waves and rendering. All numerical updates live in Slang."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import slangpy as spy
from gait import Biped

ROOT = Path(__file__).resolve().parent
DEFAULT_BACKEND = "metal" if sys.platform == "darwin" else "d3d12" if sys.platform == "win32" else "vulkan"
VIEWS = ("water", "height", "velocity", "mask", "refraction", "reflection", "fresnel")


@dataclass
class Settings:
    resolution: int = 768
    extent: float = 24.0
    wave_speed: float = 3.2
    damping: float = 1.3
    strength: float = 0.85
    water_depth: float = 0.12
    entry_gain: float = 2.0
    exit_gain: float = 0.6
    drag_gain: float = 0.25
    open_water: bool = False

    def validate(self):
        if not 32 <= self.resolution <= 2048:
            raise ValueError("resolution must be in [32, 2048]")
        for name in ("extent", "wave_speed", "damping", "strength", "water_depth", "entry_gain", "exit_gain", "drag_gain"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 8 <= self.extent <= 100 or not 0.1 <= self.wave_speed <= 12:
            raise ValueError("extent must be in [8, 100], wave speed in [0.1, 12]")
        if not 0 <= self.damping <= 8 or not 0 <= self.strength <= 8:
            raise ValueError("damping and strength must be in [0, 8]")
        if not 0 <= self.water_depth <= 0.25:
            raise ValueError("water_depth must be in [0, 0.25]")
        if not all(0 <= value <= 8 for value in (self.entry_gain, self.exit_gain, self.drag_gain)):
            raise ValueError("foot interaction gains must be in [0, 8]")

    @property
    def dx(self):
        return self.extent / self.resolution

    @property
    def timestep(self):
        # Symplectic Euler + five-point Laplacian: c*dt/dx <= 1/sqrt(2).
        return min(1.0 / 120.0, 0.6 * self.dx / (math.sqrt(2) * self.wave_speed))


def create_device(backend=DEFAULT_BACKEND, debug=False):
    return spy.Device(type=getattr(spy.DeviceType, backend), enable_debug_layers=debug,
                      compiler_options={"include_paths": [ROOT / "shaders"]})


class Water:
    def __init__(self, device, settings=None, width=1280, height=800):
        self.device = device
        self.settings = settings or Settings()
        self.settings.validate()
        self.initialize = device.create_compute_kernel(device.load_program("simulation.slang", ["initialize"]))
        self.update = device.create_compute_kernel(device.load_program("simulation.slang", ["step"]))
        self.draw = device.create_compute_kernel(device.load_program("render.slang", ["render"]))
        n = self.settings.resolution
        self.states = [self._texture(n, n, spy.Format.rgba32_float, f"water.state.{i}") for i in range(2)]
        self.mask = self._texture(n, n, spy.Format.r32_float, "water.mask")
        self.index = 0
        self.time = 0.0
        self.player = (0.0, 0.0)
        self.normal_strength = 3.2
        self.detail = 0.25
        self.zoom = 1.4
        self.follow = True
        self.foot_debug = False
        self.ior = 1.333
        self.roughness = 0.12
        self.absorption = 1.0
        self.refraction = True
        self.reflection = True
        self.floor_grid = False
        self.sun_azimuth = -170.0
        self.sun_elevation = 55.0
        self.view = 0
        self.top_down = False
        self.width = self.height = 0
        self.resize(width, height)
        self.reset()

    def _texture(self, width, height, format, label):
        return self.device.create_texture(width=width, height=height, format=format, label=label,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)

    @property
    def state(self):
        return self.states[self.index]

    def _params(self, dt=0.0, impulse=0.0, impulse_position=None):
        s = self.settings
        return dict(resolution=s.resolution, extent=s.extent, dt=dt, waveSpeed=s.wave_speed,
                    damping=s.damping, strength=s.strength, entryGain=s.entry_gain,
                    exitGain=s.exit_gain, dragGain=s.drag_gain,
                    impulsePosition=spy.float2(*(self.player if impulse_position is None else impulse_position)),
                    impulse=impulse, openWater=int(s.open_water))

    def reset(self):
        gait = Biped()
        gait.reset(depth=self.settings.water_depth, open_water=self.settings.open_water)
        self.feet = gait.feet
        encoder = self.device.create_command_encoder()
        for texture in self.states:
            self.initialize.dispatch(thread_count=[self.settings.resolution, self.settings.resolution, 1],
                vars={"g": self._params(), "g_output": texture, "g_maskOutput": self.mask},
                command_encoder=encoder)
        self.device.submit_command_buffer(encoder.finish())
        self.index, self.time = 0, 0.0

    @staticmethod
    def foot_values(foot):
        return dict(previous=spy.float3(*foot.previous), position=spy.float3(*foot.position),
                    forward=spy.float2(*foot.forward), halfSize=spy.float2(*foot.half_size),
                    halfHeight=foot.half_height, phase=foot.phase,
                    target=spy.float2(*foot.target), swinging=int(foot.swinging))

    def feet_values(self):
        if not 1 <= len(self.feet) <= 8:
            raise ValueError("the scene supports 1 to 8 feet")
        values = [self.foot_values(foot) for foot in self.feet]
        return {"g_feet": values + [values[-1]] * (8 - len(values)), "g_footCount": len(values)}

    def step(self, encoder, feet=None, dt=None, impulse=0.0, impulse_position=None):
        dt = self.settings.timestep if dt is None else dt
        if not math.isfinite(dt) or not 0 < dt <= self.settings.timestep * (1 + 1e-6):
            raise ValueError("step dt must be positive and within the configured CFL timestep")
        if feet is not None:
            self.feet = feet
        self.update.dispatch(thread_count=[self.settings.resolution, self.settings.resolution, 1],
            vars={"g": self._params(dt, impulse, impulse_position),
                  **self.feet_values(),
                  "g_state": self.state, "g_mask": self.mask, "g_output": self.states[1 - self.index]},
            command_encoder=encoder)
        self.index = 1 - self.index
        self.time += dt

    def resize(self, width, height):
        if not 1 <= width <= 8192 or not 1 <= height <= 8192:
            raise ValueError("render dimensions must be in [1, 8192]")
        if (width, height) == (self.width, self.height):
            return
        self.device.wait()
        self.display = self._texture(width, height, spy.Format.rgba32_float, "water.display")
        self.optical_data = self._texture(width, height, spy.Format.rgba32_float, "water.optical_data")
        self.width, self.height = width, height

    def render(self, encoder):
        for name, low, high in (("ior", 1.0, 1.6), ("roughness", 0.02, 0.45), ("absorption", 0.0, 4.0),
                                ("sun_azimuth", -180.0, 180.0), ("sun_elevation", 5.0, 85.0)):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        azimuth, elevation = math.radians(self.sun_azimuth), math.radians(self.sun_elevation)
        sunlight = spy.float3(math.sin(azimuth) * math.cos(elevation), math.sin(elevation),
                              math.cos(azimuth) * math.cos(elevation))
        self.draw.dispatch(thread_count=[self.width, self.height, 1],
            vars={"g_state": self.state, "g_output": self.display,
                  "g_opticalData": self.optical_data, "g_ior": self.ior,
                  "g_roughness": self.roughness, "g_absorption": self.absorption,
                  "g_refraction": int(self.refraction), "g_reflection": int(self.reflection),
                  "g_floorGrid": int(self.floor_grid), "g_sunDirection": sunlight,
                  "g_resolution": self.settings.resolution, "g_extent": self.settings.extent,
                  "g_time": self.time, "g_player": spy.float2(*self.player),
                  **self.feet_values(),
                  "g_depth": self.settings.water_depth, "g_footDebug": int(self.foot_debug),
                  "g_zoom": self.zoom, "g_cameraCenter": spy.float2(*(self.player if self.follow else (0, 0))),
                  "g_normalStrength": self.normal_strength, "g_detail": self.detail,
                  "g_view": self.view, "g_topDown": int(self.top_down),
                  "g_openWater": int(self.settings.open_water)}, command_encoder=encoder)

    def screen_to_world(self, x, y):
        """Inverse of the orthographic camera in render.slang, onto Y=0."""
        aspect = self.width / self.height
        half_width = self.settings.extent * 0.59 * max(1.0, aspect / 1.65) / self.zoom
        cosine = 1.0 if self.top_down else 1.0 / math.sqrt(1.0 + 0.78**2)
        center = np.array(self.player) if self.follow else np.zeros(2)
        return center + np.array([(x / self.width * 2.0 - 1.0) * half_width,
                                  (y / self.height * 2.0 - 1.0) * half_width / aspect / cosine])

    def read_state(self):
        self.device.wait()
        return np.array(self.state.to_numpy(), copy=True)

    def read_optics(self):
        """Fresnel weight, underwater distance, transmission ID, reflection ID.

        ID 0 is terrain/sky, 1/2 player feet, 3/4 rocks, 5..10 NPC feet.
        All channels are zero where the primary ray hits foreground geometry.
        """
        self.device.wait()
        return np.array(self.optical_data.to_numpy(), copy=True)

    def save(self, path, save_state=False):
        self.device.wait()
        rgb = self.display.to_numpy()[..., :3]
        if not np.isfinite(rgb).all():
            raise RuntimeError("nonfinite render output")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(rgb * 255.0 + 0.5, 0, 255))).save(path)
        if save_state:
            np.savez_compressed(path.with_suffix(".npz"), state=self.read_state(),
                                optics=self.read_optics(), mask=self.mask.to_numpy(),
                                extent=self.settings.extent, time=self.time)
