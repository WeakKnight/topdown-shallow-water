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
    extent: float = 32.0
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
    def edge_width(self):
        return min(3.0, self.extent * 0.125)

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
        self.scroll = device.create_compute_kernel(device.load_program("simulation.slang", ["scroll"]))
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
        self.free_camera = np.zeros(2)
        self.camera_span = 24.0  # View scale is independent of simulation coverage.
        self.domain_cells = np.zeros(2, dtype=np.int64)
        self.domain_debug = False
        self.scroll_count = self.cut_count = self.active_foot_count = 0
        self.last_shift = (0, 0)
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
        return dict(resolution=s.resolution, extent=s.extent, center=spy.float2(*self.domain_center),
                    edgeWidth=s.edge_width, dt=dt, waveSpeed=s.wave_speed,
                    damping=s.damping, strength=s.strength, entryGain=s.entry_gain,
                    exitGain=s.exit_gain, dragGain=s.drag_gain,
                    impulsePosition=spy.float2(*(self.player if impulse_position is None else impulse_position)),
                    impulse=impulse, openWater=int(s.open_water))

    @property
    def camera_center(self):
        return np.asarray(self.player if self.follow else self.free_camera, dtype=float)

    @property
    def domain_center(self):
        return self.domain_cells * self.settings.dx

    def set_camera(self, position):
        """Observe a world XZ point without moving or teleporting any actors."""
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all() or np.max(np.abs(position)) > 1e6:
            raise ValueError("camera position must be a finite XZ pair within +/-1,000,000 m")
        self.free_camera = position.copy()
        self.follow = False

    def sync_domain(self, encoder):
        """Integer reindexing only: preserves world-space history, even at dt=0."""
        cells = np.floor(self.camera_center / self.settings.dx + 0.5).astype(np.int64)
        delta = cells - self.domain_cells
        if not np.any(delta):
            return
        n = self.settings.resolution
        self.domain_cells = cells
        self.last_shift = tuple(int(v) for v in delta)
        self.scroll_count += 1
        self.cut_count += int(np.any(np.abs(delta) >= n))
        # Clamping huge cuts avoids overflowing the shader integer offset;
        # any offset >= N already makes every source texel invalid.
        offset = np.clip(delta, -n, n).astype(int)
        self.scroll.dispatch(thread_count=[n, n, 1],
            vars={"g": self._params(), "g_shift": spy.int2(*offset), "g_state": self.state,
                  "g_output": self.states[1 - self.index], "g_maskOutput": self.mask},
            command_encoder=encoder)
        self.index = 1 - self.index

    def reset(self):
        self.domain_cells = np.floor(self.camera_center / self.settings.dx + 0.5).astype(np.int64)
        self.scroll_count = self.cut_count = self.active_foot_count = 0
        self.last_shift = (0, 0)
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

    def feet_values(self, simulation=False):
        if not 1 <= len(self.feet) <= 8:
            raise ValueError("the scene supports 1 to 8 feet")
        values = []
        for foot in self.feet:
            value = self.foot_values(foot)
            if simulation:
                previous = foot.previous
                # Actor teleport is distinct from a camera cut. Never sweep a
                # foot across the world if animation poses are discontinuous.
                if np.linalg.norm(foot.position - previous) > 2.0:
                    previous = foot.position
                    value["previous"] = spy.float3(*previous)
                radius = max(foot.half_size) * 5 + self.settings.dx * 2
                low = np.minimum(previous[[0, 2]], foot.position[[0, 2]]) - radius
                high = np.maximum(previous[[0, 2]], foot.position[[0, 2]]) + radius
                if (np.any(high < self.domain_center - self.settings.extent * 0.5)
                        or np.any(low > self.domain_center + self.settings.extent * 0.5)):
                    continue
            values.append(value)
        if simulation:
            self.active_foot_count = len(values)
        padding = self.foot_values(self.feet[0])
        return {"g_feet": values + [padding] * (8 - len(values)), "g_footCount": len(values)}

    def step(self, encoder, feet=None, dt=None, impulse=0.0, impulse_position=None):
        dt = self.settings.timestep if dt is None else dt
        if not math.isfinite(dt) or not 0 < dt <= self.settings.timestep * (1 + 1e-6):
            raise ValueError("step dt must be positive and within the configured CFL timestep")
        if feet is not None:
            self.feet = feet
        self.sync_domain(encoder)
        self.update.dispatch(thread_count=[self.settings.resolution, self.settings.resolution, 1],
            vars={"g": self._params(dt, impulse, impulse_position),
                  **self.feet_values(simulation=True),
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
        self.sync_domain(encoder)
        self.feet_values(simulation=True)  # Keep the UI count current when paused or cutting cameras.
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
                  "g_domainCenter": spy.float2(*self.domain_center), "g_edgeWidth": self.settings.edge_width,
                  "g_domainDebug": int(self.domain_debug), "g_cameraSpan": self.camera_span,
                  "g_time": self.time, "g_player": spy.float2(*self.player),
                  **self.feet_values(),
                  "g_depth": self.settings.water_depth, "g_footDebug": int(self.foot_debug),
                  "g_zoom": self.zoom, "g_cameraCenter": spy.float2(*self.camera_center),
                  "g_normalStrength": self.normal_strength, "g_detail": self.detail,
                  "g_view": self.view, "g_topDown": int(self.top_down),
                  "g_openWater": int(self.settings.open_water)}, command_encoder=encoder)

    def screen_to_world(self, x, y):
        """Inverse of the orthographic camera in render.slang, onto Y=0."""
        aspect = self.width / self.height
        half_width = self.camera_span * 0.59 * max(1.0, aspect / 1.65) / self.zoom
        cosine = 1.0 if self.top_down else 1.0 / math.sqrt(1.0 + 0.78**2)
        center = self.camera_center
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
                                extent=self.settings.extent, time=self.time, domain_center=self.domain_center,
                                camera_center=self.camera_center, dx=self.settings.dx)
