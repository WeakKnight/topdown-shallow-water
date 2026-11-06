"""Interactive MOBA water prototype. Run with ./run.sh or python app.py."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import slangpy as spy

from water import DEFAULT_BACKEND, ROOT, VIEWS, Settings, Water, create_device
from gait import Biped
from crowd import Wanderer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Slang / SlangPy interactive height-field water")
    p.add_argument("--backend", choices=("metal", "vulkan", "d3d12"), default=DEFAULT_BACKEND)
    p.add_argument("--headless", action="store_true", help="Run deterministic 60 Hz demo and save PNG")
    p.add_argument("--frames", type=int, default=360)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--extent", type=float, default=24.0, help="Square world-space RT width in meters")
    p.add_argument("--wave-speed", type=float, default=3.2)
    p.add_argument("--damping", type=float, default=1.3)
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--water-depth", type=float, default=0.12)
    p.add_argument("--entry-strength", type=float, default=2.0)
    p.add_argument("--exit-strength", type=float, default=0.6)
    p.add_argument("--drag-strength", type=float, default=0.25)
    p.add_argument("--cadence", type=float, default=2.0, help="Combined left/right steps per second at walking speed")
    p.add_argument("--lift-height", type=float, default=0.28, help="Sole lift above the ground, in meters")
    p.add_argument("--zoom", type=float, default=1.4)
    p.add_argument("--npcs", type=int, choices=range(4), default=3, help="Number of autonomous wandering bipeds")
    p.add_argument("--video", type=Path, help="Export rendered frames as H.264 MP4, without UI")
    p.add_argument("--seconds", type=float, default=20.0, help="Video duration")
    p.add_argument("--fps", type=int, choices=(30, 60), default=60)
    p.add_argument("--no-follow", action="store_true")
    p.add_argument("--foot-debug", action="store_true")
    p.add_argument("--open-water", action="store_true", help="All-wet domain with an absorbing outer border")
    p.add_argument("--view", choices=VIEWS, default="water")
    p.add_argument("--ior", type=float, default=1.333)
    p.add_argument("--roughness", type=float, default=0.12)
    p.add_argument("--absorption", type=float, default=1.0)
    p.add_argument("--sun-azimuth", type=float, default=-170.0)
    p.add_argument("--sun-elevation", type=float, default=55.0)
    p.add_argument("--no-refraction", action="store_true", help="Disable ray bending; keep absorption")
    p.add_argument("--no-reflection", action="store_true", help="Display full transmission without reflected light")
    p.add_argument("--floor-grid", action="store_true", help="25 cm floor grid to inspect refraction")
    p.add_argument("--top-down", action="store_true")
    p.add_argument("--no-demo", action="store_true", help="Start with both feet planted")
    p.add_argument("--no-detail", action="store_true", help="Disable decorative normal waves and caustics")
    p.add_argument("--output", type=Path, default=ROOT / "output" / "water.png")
    p.add_argument("--save-state", action="store_true", help="Also save raw state/mask as NPZ")
    p.add_argument("--debug", action="store_true", help="Enable GPU validation layers")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--window-frames", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        settings_from_args(args).validate()
    except ValueError as error:
        p.error(str(error))
    if not 1 <= args.width <= 8192 or not 1 <= args.height <= 8192:
        p.error("width and height must be in [1, 8192]")
    if args.frames < 1 or args.window_frames < 0:
        p.error("frames must be positive; window-frames must be nonnegative")
    for name, low, high in (("cadence", 0.6, 4.0), ("lift_height", 0.04, 0.6), ("zoom", 1.0, 5.0),
                            ("ior", 1.0, 1.6), ("roughness", 0.02, 0.45), ("absorption", 0.0, 4.0),
                            ("sun_azimuth", -180.0, 180.0), ("sun_elevation", 5.0, 85.0)):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            p.error(f"{name} must be in [{low}, {high}]")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
        p.error("seconds must be in (0, 600]")
    if args.video:
        if args.video.suffix.lower() != ".mp4" or args.width % 2 or args.height % 2:
            p.error("video requires an .mp4 path and even width/height")
        args.video = args.video.expanduser().resolve()
    # macOS window initialization may change cwd to Python.app's Resources.
    args.output = args.output.expanduser().resolve()
    return args


def settings_from_args(args):
    return Settings(resolution=args.resolution, extent=args.extent, wave_speed=args.wave_speed,
                    damping=args.damping, strength=args.strength, water_depth=args.water_depth,
                    entry_gain=args.entry_strength, exit_gain=args.exit_strength, drag_gain=args.drag_strength,
                    open_water=args.open_water)


class App:
    def __init__(self, args):
        self.args = args
        self.device = create_device(args.backend, args.debug)
        self.water = Water(self.device, settings_from_args(args), args.width, args.height)
        self.water.view = VIEWS.index(args.view)
        for name in ("ior", "roughness", "absorption", "sun_azimuth", "sun_elevation", "floor_grid"):
            setattr(self.water, name, getattr(args, name))
        self.water.refraction = not args.no_refraction
        self.water.reflection = not args.no_reflection
        self.water.top_down = args.top_down
        self.water.detail = 0.0 if args.no_detail else 0.25
        self.water.zoom, self.water.follow = args.zoom, not args.no_follow
        self.water.foot_debug = args.foot_debug
        self.gait = Biped(args.cadence, args.lift_height)
        self.gait.reset(depth=args.water_depth, open_water=args.open_water)
        self.npcs = [Wanderer(i, args.extent, args.water_depth, args.open_water) for i in range(args.npcs)]
        self.water.feet = self.gait.feet + [foot for npc in self.npcs for foot in npc.gait.feet]
        self.position = np.zeros(2)
        self.target = None
        self.auto = not args.no_demo
        self.auto_phase = 0.0
        self.paused = False
        self.accumulator = 0.0
        self.impulse = 0.0
        self.keys = set()
        self.capture_requested = False
        self.frame_count = 0

    def reset(self):
        self.water.reset()
        self.position[:] = 0
        self.water.player = (0.0, 0.0)
        self.gait.reset(depth=self.water.settings.water_depth, open_water=self.water.settings.open_water)
        for npc in self.npcs:
            npc.reset(self.water.settings.water_depth, self.water.settings.open_water)
        self.water.feet = self.gait.feet + [foot for npc in self.npcs for foot in npc.gait.feet]
        self.target = None
        self.auto_phase = self.accumulator = self.impulse = 0.0

    def advance(self, encoder, elapsed):
        if self.paused:
            return
        # Cap wall-time debt after dragging/resizing the window (no spiral of death).
        self.accumulator += min(elapsed, 0.1)
        dt = self.water.settings.timestep
        while self.accumulator + 1e-12 >= dt:
            previous = self.position.copy()
            k = spy.KeyCode
            movement = np.array([int(k.d in self.keys) - int(k.a in self.keys),
                                 int(k.s in self.keys) - int(k.w in self.keys)], dtype=float)
            speed = 3.8 if k.left_shift in self.keys else 1.4
            if np.any(movement):
                self.auto, self.target = False, None
                self.position += movement / np.linalg.norm(movement) * speed * dt
            elif self.auto:
                self.auto_phase += dt * 0.22
                target = np.array([5.4 * math.sin(self.auto_phase), 2.6 * math.sin(self.auto_phase * 2)])
                # Bounded motion also prevents teleporting when demo is toggled on.
                delta = target - self.position
                self.position += delta * min(1.0, 1.6 * dt / max(np.linalg.norm(delta), 1e-8))
            elif self.target is not None:
                delta = self.target - self.position
                self.position += delta * min(1.0, speed * dt / max(np.linalg.norm(delta), 1e-8))
            limit = self.water.settings.extent * 0.5 - 1.2
            self.position = np.clip(self.position, -limit, limit)
            feet = self.gait.update(self.position, (self.position - previous) / dt, dt,
                                   self.water.settings.water_depth, self.water.settings.open_water)
            feet = list(feet)
            for npc in self.npcs:
                feet.extend(npc.update(dt, self.water.settings.water_depth, self.water.settings.open_water))
            self.water.player = tuple(self.position)
            self.water.step(encoder, feet, dt, impulse=self.impulse)
            self.impulse = 0.0
            self.accumulator = max(0.0, self.accumulator - dt)

    def run_headless(self):
        start = time.perf_counter()
        for i in range(self.args.frames):
            encoder = self.device.create_command_encoder()
            self.advance(encoder, 1.0 / 60.0)
            if i == self.args.frames - 1:
                self.water.render(encoder)
            self.device.submit_command_buffer(encoder.finish())
            if (i + 1) % 32 == 0:
                self.device.wait()
        self.water.save(self.args.output, self.args.save_state)
        state = self.water.read_state()
        if not np.isfinite(state).all():
            raise RuntimeError("nonfinite simulation state")
        print(f"Saved {self.args.output.resolve()} | t={self.water.time:.3f}s | "
              f"height=[{state[..., 0].min():.5f}, {state[..., 0].max():.5f}] m | "
              f"{time.perf_counter() - start:.2f}s", flush=True)

    def run_video(self):
        from recording import record
        record(self)

    def setup_ui(self):
        # SlangPy scales its font/style by display DPI, but window/input sizes
        # remain logical pixels. Use matching physical UI coordinates throughout.
        self.ui_scale = spy.platform.display_scale_factor()
        self.ui = spy.ui.Context(self.device)
        panel = spy.ui.Window(self.ui.screen, "Water lab", spy.float2(16, 16) * self.ui_scale,
                              spy.float2(310, min(750, self.window.height - 32)) * self.ui_scale)
        spy.ui.Text(panel, "SLANG / BIPED WATER")
        spy.ui.Text(panel, "Player: orange + blue")
        spy.ui.Text(panel, f"NPCs: {len(self.npcs)} | green / purple / gold")
        self.stats = spy.ui.Text(panel, "")
        self.status = spy.ui.Text(panel, "")
        self.foot_status = spy.ui.Text(panel, "")
        spy.ui.Button(panel, "Demo on / off [Tab]", callback=self.toggle_demo)
        spy.ui.Button(panel, "Pause / resume [P]", callback=lambda: setattr(self, "paused", not self.paused))
        spy.ui.Button(panel, "Splash [Space]", callback=lambda: setattr(self, "impulse", 2.5))
        spy.ui.Button(panel, "Reset [R]", callback=self.reset)
        spy.ui.Button(panel, "Cycle debug view [V]", callback=self.cycle_view)
        spy.ui.Button(panel, "Change camera [T]",
                      callback=lambda: setattr(self.water, "top_down", not self.water.top_down))
        spy.ui.Button(panel, "Foot outlines + targets [B]",
                      callback=lambda: setattr(self.water, "foot_debug", not self.water.foot_debug))
        spy.ui.Button(panel, "Follow camera [F]",
                      callback=lambda: setattr(self.water, "follow", not self.water.follow))
        for label, attr, minimum, maximum in (("Wave speed", "wave_speed", 0.5, 8.0),
                ("Damping", "damping", 0.0, 3.0), ("Interaction", "strength", 0.0, 5.0),
                ("Water depth (m)", "water_depth", 0.0, 0.25),
                ("Entry impulse", "entry_gain", 0.0, 6.0), ("Exit refill", "exit_gain", 0.0, 3.0),
                ("Wet drag", "drag_gain", 0.0, 2.0)):
            spy.ui.SliderFloat(panel, label, min=minimum, max=maximum,
                value=getattr(self.water.settings, attr),
                callback=lambda value, attr=attr: setattr(self.water.settings, attr, value))
        for label, attr, minimum, maximum in (("Steps / second", "cadence", 0.6, 4.0),
                                              ("Foot lift (m)", "lift_height", 0.04, 0.6)):
            spy.ui.SliderFloat(panel, label, min=minimum, max=maximum, value=getattr(self.gait, attr),
                              callback=lambda value, attr=attr: setattr(self.gait, attr, value))
        spy.ui.SliderFloat(panel, "Zoom", min=1.0, max=5.0, value=self.water.zoom,
                          callback=lambda value: setattr(self.water, "zoom", value))
        spy.ui.SliderFloat(panel, "Normal strength", min=0.0, max=5.0, value=self.water.normal_strength,
                          callback=lambda value: setattr(self.water, "normal_strength", value))
        spy.ui.SliderFloat(panel, "Surface detail", min=0.0, max=1.0, value=self.water.detail,
                          callback=lambda value: setattr(self.water, "detail", value))
        spy.ui.Button(panel, "Save PNG [F2]", callback=lambda: setattr(self, "capture_requested", True))
        spy.ui.Text(panel, "WASD: move   Shift: dash")
        spy.ui.Text(panel, "Click: move   Wheel: zoom   Esc: quit")
        self.optics_panel = spy.ui.Window(self.ui.screen, "Water optics",
            spy.float2(max(340, self.window.width - 302), 16) * self.ui_scale,
            spy.float2(286, 440) * self.ui_scale)
        optics = self.optics_panel
        spy.ui.Text(optics, "WORLD-SPACE OPTICS")
        self.optics_status = spy.ui.Text(optics, "")
        spy.ui.Button(optics, "Refraction on / off [I]", callback=self.toggle_refraction)
        spy.ui.Button(optics, "Reflection on / off [O]", callback=self.toggle_reflection)
        spy.ui.Button(optics, "Floor grid on / off [G]", callback=self.toggle_grid)
        for label, attr, minimum, maximum in (("Water IOR", "ior", 1.0, 1.6),
                ("Roughness", "roughness", 0.02, 0.45), ("Absorption", "absorption", 0.0, 4.0),
                ("Sun azimuth", "sun_azimuth", -180.0, 180.0), ("Sun elevation", "sun_elevation", 5.0, 85.0)):
            spy.ui.SliderFloat(optics, label, min=minimum, max=maximum, value=getattr(self.water, attr),
                              callback=lambda value, attr=attr: setattr(self.water, attr, value))
        spy.ui.Text(optics, "P: freeze for A/B comparison")
        spy.ui.Text(optics, "V: transmission / reflection / F")
        spy.ui.Text(optics, "Fresnel view: raw 0..1 weights")

    def toggle_refraction(self):
        self.water.refraction = not self.water.refraction

    def toggle_reflection(self):
        self.water.reflection = not self.water.reflection

    def toggle_grid(self):
        self.water.floor_grid = not self.water.floor_grid

    def toggle_demo(self):
        self.auto = not self.auto
        self.target = None

    def cycle_view(self):
        self.water.view = (self.water.view + 1) % len(VIEWS)

    def keyboard_event(self, event):
        consumed = self.ui.handle_keyboard_event(event)
        if event.is_key_release():
            self.keys.discard(event.key)
        if not event.is_key_press() or consumed:
            return
        k = spy.KeyCode
        self.keys.add(event.key)
        if event.key == k.escape:
            self.window.close()
        elif event.key == k.r:
            self.reset()
        elif event.key == k.space:
            self.impulse = 2.5
        elif event.key == k.p:
            self.paused = not self.paused
        elif event.key == k.tab:
            self.toggle_demo()
        elif event.key == k.v:
            self.cycle_view()
        elif event.key == k.t:
            self.water.top_down = not self.water.top_down
        elif event.key == k.b:
            self.water.foot_debug = not self.water.foot_debug
        elif event.key == k.f:
            self.water.follow = not self.water.follow
        elif event.key == k.i:
            self.toggle_refraction()
        elif event.key == k.o:
            self.toggle_reflection()
        elif event.key == k.g:
            self.toggle_grid()
        elif event.key == k.f2:
            self.capture_requested = True

    def mouse_event(self, event):
        ui_event = spy.MouseEvent()
        ui_event.type, ui_event.button, ui_event.mods = event.type, event.button, event.mods
        ui_event.pos, ui_event.scroll = event.pos * self.ui_scale, event.scroll
        if self.ui.handle_mouse_event(ui_event):
            return
        if event.is_scroll():
            self.water.zoom = float(np.clip(self.water.zoom * math.exp(event.scroll.y * 0.1), 1, 5))
        if event.is_button_down() and event.button == spy.MouseButton.left:
            # Window events are logical pixels; rendering may use a HiDPI surface.
            x = event.pos.x * self.water.width / self.window.width
            y = event.pos.y * self.water.height / self.window.height
            self.target = self.water.screen_to_world(x, y)
            self.auto = False

    def resize(self, width, height):
        self.device.wait()
        if width > 0 and height > 0:
            render_width = round(width * self.ui_scale)
            render_height = round(height * self.ui_scale)
            self.surface.configure(width=render_width, height=render_height, format=self.surface_format,
                                   vsync=not self.args.no_vsync)
            self.water.resize(render_width, render_height)
        else:
            self.surface.unconfigure()

    def run_interactive(self):
        mode = spy.WindowMode.minimized if self.args.window_frames else spy.WindowMode.normal
        self.window = spy.Window(width=self.args.width, height=self.args.height,
                                 title="Biped Water | Slang + SlangPy", resizable=True, mode=mode)
        self.ui_scale = spy.platform.display_scale_factor()
        self.surface = self.device.create_surface(self.window)
        self.surface_format = next((fmt for fmt in (spy.Format.rgba8_unorm, spy.Format.bgra8_unorm)
                                    if fmt in self.surface.info.formats), None)
        if self.surface_format is None:
            raise RuntimeError("An 8-bit UNORM presentation surface is required")
        self.resize(self.args.width, self.args.height)
        self.setup_ui()
        self.window.on_keyboard_event = self.keyboard_event
        self.window.on_mouse_event = self.mouse_event
        self.window.on_resize = self.resize
        previous = time.perf_counter()
        while not self.window.should_close():
            now = time.perf_counter()
            elapsed, previous = now - previous, now
            self.window.process_events()
            if self.window.should_close():
                break
            if not self.surface.config:
                time.sleep(0.02)
                continue
            texture = self.surface.acquire_next_image()
            if texture is None:
                time.sleep(0.01)
                continue
            self.water.resize(texture.width, texture.height)
            encoder = self.device.create_command_encoder()
            self.advance(encoder, 1 / 60 if self.args.window_frames else elapsed)
            self.water.render(encoder)
            encoder.blit(texture, self.water.display)
            s = self.water.settings
            self.stats.text = f"{s.resolution} x {s.resolution} | {s.dx * 100:.1f} cm/texel\n{elapsed * 1000:.1f} ms / frame"
            view = VIEWS[self.water.view].upper()
            self.status.text = f"{view} | {'PAUSED' if self.paused else 'DEMO' if self.auto else 'MANUAL'} | t={self.water.time:.1f}s"
            labels = []
            for name, foot in zip(("L", "R"), self.gait.feet):
                wet = foot.immersion > 0 and (s.open_water or self._foot_in_water(foot))
                labels.append(f"{name}: {'SWING' if foot.swinging else 'PLANTED'} / {'WET' if wet else 'DRY'}")
            self.foot_status.text = "\n".join(labels)
            self.optics_status.text = f"Refraction: {'ON' if self.water.refraction else 'OFF'}\nReflection: {'ON' if self.water.reflection else 'OFF'}"
            self.ui.begin_frame(round(self.window.width * self.ui_scale),
                                round(self.window.height * self.ui_scale))
            self.ui.end_frame(texture, encoder)
            self.device.submit_command_buffer(encoder.finish())
            del texture
            self.surface.present()
            if self.capture_requested:
                self.water.save(self.args.output, self.args.save_state)
                self.capture_requested = False
                print(f"Saved {self.args.output.resolve()}", flush=True)
            self.frame_count += 1
            if self.args.window_frames and self.frame_count >= self.args.window_frames:
                self.window.close()
        self.device.wait()
        self.surface.unconfigure()

    @staticmethod
    def _foot_in_water(foot):
        from gait import water_margin
        return water_margin(foot.position[[0, 2]]) > 0

    def run(self):
        s = self.water.settings
        print(f"{self.args.backend} | RT {s.resolution}² / {s.extent:g} m | "
              f"dx={s.dx:.4f} m | dt<={s.timestep:.5f} s", flush=True)
        try:
            if self.args.video:
                self.run_video()
            elif self.args.headless:
                self.run_headless()
            else:
                self.run_interactive()
        finally:
            self.device.wait()


if __name__ == "__main__":
    App(parse_args()).run()
