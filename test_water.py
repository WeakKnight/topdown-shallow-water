"""Numerical and rendering checks against real GPU dispatches, no mocked shaders."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from app import App, parse_args
from water import Settings, Water, create_device
from gait import Biped
from crowd import Wanderer


class ConfigurationTests(unittest.TestCase):
    def test_cfl_at_all_ui_extremes(self):
        for resolution in (32, 384, 2048):
            for speed in (0.1, 3.2, 12.0):
                settings = Settings(resolution=resolution, wave_speed=speed)
                settings.validate()
                self.assertLess(settings.wave_speed * settings.timestep / settings.dx, 1 / math.sqrt(2))

    def test_invalid_cli(self):
        for arguments in (["--resolution", "0"], ["--damping", "-1"], ["--wave-speed", "nan"],
                          ["--extent", "inf"], ["--water-depth", "-1"], ["--width", "0"], ["--frames", "0"],
                          ["--cadence", "0"], ["--lift-height", "nan"], ["--zoom", "0"],
                          ["--ior", "0.9"], ["--roughness", "0"], ["--absorption", "nan"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments)


class GaitChecks(unittest.TestCase):
    def test_support_feet_lock_and_landings_alternate(self):
        gait = Biped()
        root = np.zeros(2)
        sequence = []
        dry_swing = False
        for _ in range(600):
            old = [(f.position.copy(), f.swinging) for f in gait.feet]
            counts = gait.landings.copy()
            root += np.array([1.4, 0]) / 120
            gait.update(root, (1.4, 0), 1 / 120)
            for i, foot in enumerate(gait.feet):
                if not old[i][1] and not foot.swinging:
                    np.testing.assert_array_equal(foot.position[[0, 2]], old[i][0][[0, 2]])
                if gait.landings[i] != counts[i]:
                    sequence.append(i)
                dry_swing |= foot.swinging and foot.immersion == 0
        self.assertGreater(len(sequence), 6)
        self.assertTrue(all(a != b for a, b in zip(sequence, sequence[1:])))
        self.assertTrue(dry_swing)

    def test_stop_finishes_step_and_settles_both_feet(self):
        gait = Biped()
        root = np.zeros(2)
        for _ in range(67):
            root += np.array([0, -1.4]) / 120
            gait.update(root, (0, -1.4), 1 / 120)
        for _ in range(360):
            gait.update(root, (0, 0), 1 / 120)
        self.assertTrue(all(not f.swinging for f in gait.feet))
        for i, foot in enumerate(gait.feet):
            self.assertLess(np.linalg.norm(foot.position[[0, 2]] - root - [(-0.19 if i == 0 else 0.19), 0]), 0.14)
        frozen = [f.position.copy() for f in gait.feet]
        counts = gait.landings.copy()
        for _ in range(240):
            gait.update(root, (0, 0), 1 / 120)
        self.assertEqual(counts, gait.landings)
        for foot, expected in zip(gait.feet, frozen):
            np.testing.assert_array_equal(foot.position, expected)

    def test_sprint_and_abrupt_turns_are_finite(self):
        gait = Biped(cadence=4.0, lift_height=0.5)
        root = np.zeros(2)
        for i in range(1000):
            velocity = np.array([3.8 if (i // 100) % 2 else -3.8, 0])
            root += velocity / 240
            gait.update(root, velocity, 1 / 240)
            for foot in gait.feet:
                self.assertTrue(np.isfinite(foot.position).all())
                self.assertLess(np.linalg.norm(foot.position[[0, 2]] - root), 2.0)


class GPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device()
        cls.water = Water(cls.device, Settings(resolution=128), width=97, height=65)

    @classmethod
    def tearDownClass(cls):
        cls.device.wait()

    def setUp(self):
        self.water.settings = Settings(resolution=128)
        self.water.player = (0.0, 0.0)
        self.water.reset()
        self.gait = Biped()
        self.root = np.zeros(2)

    def steps(self, count, motion=None, impulse=0.0):
        dt = self.water.settings.timestep
        for i in range(count):
            encoder = self.device.create_command_encoder()
            current = self.root if motion is None else np.array(motion(self.water.time + dt))
            feet = self.gait.update(current, (current - self.root) / dt, dt,
                                    self.water.settings.water_depth, self.water.settings.open_water)
            self.root = np.array(current)
            self.water.step(encoder, feet, impulse=impulse if i == 0 else 0)
            self.device.submit_command_buffer(encoder.finish())
            if (i + 1) % 64 == 0:
                self.device.wait()
        result = self.water.read_state()
        self.assertTrue(np.isfinite(result).all())
        return result

    def test_rest_is_exactly_still(self):
        np.testing.assert_array_equal(self.steps(120), 0)

    def test_pulse_propagates_symmetrically_and_loses_energy(self):
        self.water.settings.open_water = True
        self.water.settings.damping = 2.0
        self.water.reset()
        first = self.steps(1, impulse=2.5)
        propagated = self.steps(70)
        dx = self.water.settings.dx
        coordinates = (np.arange(128) + 0.5 - 64) * dx
        x, z = np.meshgrid(coordinates, coordinates)
        ring = (x * x + z * z > 1.5**2) & (x * x + z * z < 2.5**2)
        self.assertGreater(np.max(np.abs(propagated[..., 0][ring])), 0.005)
        np.testing.assert_allclose(propagated[..., 0], propagated[::-1, :, 0], atol=2e-6)
        np.testing.assert_allclose(propagated[..., 0], propagated[:, ::-1, 0], atol=2e-6)

        def energy(state):
            h, v = state[..., 0], state[..., 1]
            return np.sum(v * v) + (self.water.settings.wave_speed / dx)**2 * (
                np.sum(np.diff(h, axis=0)**2) + np.sum(np.diff(h, axis=1)**2))

        final = self.steps(240)
        self.assertLess(energy(final), energy(first) * 0.05)

    def test_walking_produces_waves_and_dry_cells_stay_zero(self):
        result = self.steps(360, motion=lambda t: (min(10.5, 3.5 * t), 0))
        mask = self.water.mask.to_numpy().reshape(128, 128)
        np.testing.assert_array_equal(result[mask < 0.5], 0)
        self.assertGreater(np.max(np.abs(result[..., 0])), 0.002)
        self.assertLess(np.max(np.abs(result[..., 0])), 1.0)

    def test_speed_and_step_changes_remain_finite(self):
        for speed in (0.5, 8.0, 3.2):
            self.water.settings.wave_speed = speed
            result = self.steps(120, motion=lambda t: (2 * math.sin(t * 4), 0))
            self.assertLess(np.max(np.abs(result[..., 0])), 2.0)

    def test_render_modes_resize_and_export(self):
        state = self.steps(40, impulse=2.5)
        self.water.resize(113, 71)  # Deliberately not divisible by threadgroup size.
        np.testing.assert_array_equal(self.water.read_state(), state)
        images = []
        for mode in range(4):
            self.water.view = mode
            encoder = self.device.create_command_encoder()
            self.water.render(encoder)
            self.device.submit_command_buffer(encoder.finish())
            self.device.wait()
            image = self.water.display.to_numpy()
            self.assertEqual(image.shape, (71, 113, 4))
            self.assertTrue(np.isfinite(image).all())
            self.assertGreaterEqual(image.min(), 0)
            self.assertLessEqual(image.max(), 1)
            images.append(image.copy())
        self.assertGreater(np.mean(np.abs(images[0] - images[1])), 0.05)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "water.png"
            self.water.save(path, save_state=True)
            with Image.open(path) as image:
                self.assertEqual(image.size, (113, 71))
            with np.load(path.with_suffix(".npz")) as data:
                np.testing.assert_array_equal(data["state"], state)

    def test_simulation_is_independent_of_render_frame_rate(self):
        # Exercise the actual app accumulator at 30 Hz and 144 Hz, for one second.
        app = App.__new__(App)
        app.water = self.water
        app.device = self.device
        app.position = np.zeros(2)
        app.gait = Biped()
        app.npcs = [Wanderer(i) for i in range(3)]
        app.keys = set()
        app.auto = True
        app.paused = False
        outputs = []
        for fps in (30, 144):
            app.reset()
            for _ in range(fps):
                encoder = self.device.create_command_encoder()
                app.advance(encoder, 1 / fps)
                self.device.submit_command_buffer(encoder.finish())
            outputs.append(self.water.read_state())
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7)

    def test_airborne_motion_does_not_touch_water(self):
        feet = Biped().feet
        for foot in feet:
            foot.previous[:] = [-1, 0.5, 0]
            foot.position[:] = [1, 0.7, 0]
        encoder = self.device.create_command_encoder()
        self.water.step(encoder, feet)
        self.device.submit_command_buffer(encoder.finish())
        np.testing.assert_array_equal(self.water.read_state(), 0)

    def test_entry_depresses_center_and_exit_refills(self):
        for entering in (True, False):
            self.water.reset()
            feet = Biped().feet
            center = self.water.settings.dx * 0.5
            feet[0].previous[:] = [center, 0.10 if entering else 0.08, center]
            feet[0].position[:] = [center, 0.08 if entering else 0.10, center]
            encoder = self.device.create_command_encoder()
            self.water.step(encoder, feet)
            self.device.submit_command_buffer(encoder.finish())
            velocity = self.water.read_state()[64, 64, 1]
            self.assertLess(velocity, 0) if entering else self.assertGreater(velocity, 0)

    def test_stopping_removes_forcing_and_waves_decay(self):
        self.water.settings.damping = 2.0
        self.steps(120, motion=lambda t: (t * 1.4, 0))
        settled = self.steps(240)
        self.assertTrue(all(not foot.swinging for foot in self.gait.feet))

        def energy(state):
            h, v = state[..., 0], state[..., 1]
            return np.sum(v * v) + (self.water.settings.wave_speed / self.water.settings.dx)**2 * (
                np.sum(np.diff(h, axis=0)**2) + np.sum(np.diff(h, axis=1)**2))

        final = self.steps(360)
        self.assertLess(energy(final), energy(settled) * 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
