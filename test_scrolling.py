"""Real GPU checks for camera-relative storage and world-space wave history."""
import unittest

import numpy as np
import slangpy as spy

from app import App, parse_args
from gait import Biped, water_margin
from water import Settings, Water, create_device


class ScrollingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device()

    def make_water(self, n=64, extent=16, open_water=True):
        water = Water(self.device, Settings(resolution=n, extent=extent, open_water=open_water), 129, 97)
        water.set_camera((0, 0))
        return water

    def submit(self, action):
        encoder = self.device.create_command_encoder()
        action(encoder)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()

    def test_integer_scroll_preserves_all_channels_and_clears_new_strips(self):
        w = self.make_water()
        expected = np.random.default_rng(11).normal(size=(64, 64, 4)).astype(np.float32)
        w.state.copy_from_numpy(expected)
        z, x = np.mgrid[:64, :64]
        for dx, dz in ((3, -2), (-7, 5), (0, -1), (63, 0), (-64, 0), (500, -400)):
            source_x, source_z = x + dx, z + dz
            valid = (source_x >= 0) & (source_x < 64) & (source_z >= 0) & (source_z < 64)
            shifted = np.zeros_like(expected)
            shifted[valid] = expected[source_z[valid], source_x[valid]]
            w.set_camera(w.domain_center + np.array([dx, dz]) * w.settings.dx)
            self.submit(w.sync_domain)
            np.testing.assert_array_equal(w.read_state(), shifted)
            np.testing.assert_array_equal(w.mask.to_numpy(), 1)
            expected = shifted
        self.assertEqual(w.time, 0)
        self.assertEqual(w.cut_count, 2)

    def test_subcell_camera_moves_do_not_resample_or_change_history(self):
        w = self.make_water()
        state = np.random.default_rng(8).normal(size=(64, 64, 4)).astype(np.float32)
        w.state.copy_from_numpy(state)
        for x in (0.1, -0.49, 0.49, -0.2, 0):
            w.set_camera(np.array([x, -x]) * w.settings.dx)
            self.submit(w.sync_domain)
        self.assertEqual(w.scroll_count, 0)
        np.testing.assert_array_equal(w.read_state(), state)

    def test_shore_mask_is_world_aligned_after_negative_and_large_moves(self):
        w = self.make_water(open_water=False)
        for center in ((9.7, 0), (-4, 2), (-80, 4), (0, 0)):
            w.set_camera(center)
            self.submit(w.sync_domain)
            mask = w.mask.to_numpy().reshape(64, 64)
            expected = np.array([[water_margin(w.domain_center + (np.array([x, z]) + 0.5 - 32) * w.settings.dx) > 0
                                  for x in range(64)] for z in range(64)])
            np.testing.assert_array_equal(mask, expected)
            np.testing.assert_array_equal(w.read_state(), 0)

    def test_moving_window_matches_larger_fixed_simulation_interior(self):
        local, reference = self.make_water(), self.make_water(128, 32)
        # Both have exactly the same dx, time step and world lattice.
        for w in (local, reference):
            for foot in w.feet:
                foot.previous[:] = foot.position[:] = (100, 1, 100)
        for i in range(100):
            local.set_camera((min(i, 60) / 40, -min(i, 40) / 40))
            self.submit(lambda enc: (local.step(enc, impulse=2 if i == 0 else 0, impulse_position=(0, 0)),
                                     reference.step(enc, impulse=2 if i == 0 else 0, impulse_position=(0, 0))))
        state, large = local.read_state(), reference.read_state()
        cx, cz = local.domain_cells
        # Compare a central 8m region, away from both absorbing boundaries.
        np.testing.assert_allclose(state[16:48, 16:48], large[48+cz:80+cz, 48+cx:80+cx], atol=2e-6)
        self.assertGreater(np.max(np.abs(state[..., 0])), 1e-4)

    def test_water_and_camera_footprint_are_independent_of_simulation_extent(self):
        a, b = self.make_water(64, 8), self.make_water(64, 32)
        images = []
        for w in (a, b):
            w.zoom = 0.5  # Most of this view lies outside the smaller simulation.
            w.set_camera((80, -30))
            self.submit(w.render)
            self.assertTrue(np.all(w.read_optics()[..., 1] > 0))
            images.append(w.display.to_numpy().copy())
        np.testing.assert_allclose(a.screen_to_world(0, 0), b.screen_to_world(0, 0))
        np.testing.assert_allclose(images[0], images[1], atol=1e-6)

    def test_boundary_does_not_wrap_to_opposite_side(self):
        w = self.make_water()
        state = np.zeros((64, 64, 4), np.float32)
        state[30:34, 1, 0] = 0.01
        w.state.copy_from_numpy(state)
        for _ in range(8):
            self.submit(w.step)
        state = w.read_state()
        self.assertTrue(np.isfinite(state).all())
        np.testing.assert_array_equal(state[:, -8:], 0)

    def test_remote_npc_forces_water_while_player_is_culled(self):
        w = self.make_water()
        player, npc = Biped(), Biped()
        npc.reset((60, 0), open_water=True)
        for foot in npc.feet:
            foot.previous[1] = 0.3
        w.player = (0, 0)
        w.set_camera((60, 0))
        w.feet = player.feet + npc.feet
        self.submit(w.step)
        self.assertEqual(w.active_foot_count, 2)
        self.assertGreater(np.abs(w.read_state()[..., 1]).max(), 0.01)
        # A camera cut must not modify the scene's current/previous poses.
        poses = [(f.position.copy(), f.previous.copy()) for f in w.feet]
        w.set_camera((-60, 0))
        self.submit(w.step)
        self.assertEqual(w.active_foot_count, 0)
        np.testing.assert_array_equal(w.read_state(), 0)
        for foot, (position, previous) in zip(w.feet, poses):
            np.testing.assert_array_equal(foot.position, position)
            np.testing.assert_array_equal(foot.previous, previous)

    def test_actor_teleport_does_not_inject_a_swept_foot_trail(self):
        w = self.make_water()
        for foot in w.feet:
            foot.previous[:] = (-40, 0.5, 0)
        self.submit(w.step)
        np.testing.assert_array_equal(w.read_state(), 0)

    def test_paused_camera_pan_and_cut_leave_actors_and_time_unchanged(self):
        app = App(parse_args(["--resolution", "64", "--width", "96", "--height", "64", "--open-water", "--no-demo"]))
        w = app.water
        app.impulse = 2
        enc = app.device.create_command_encoder()
        app.advance(enc, 1 / 60)
        app.device.submit_command_buffer(enc.finish())
        self.assertGreater(np.abs(w.read_state()).max(), 0)
        app.paused = True
        app.keys = {spy.KeyCode.right}
        roots = [app.position.copy()] + [npc.position.copy() for npc in app.npcs]
        poses = [foot.position.copy() for foot in w.feet]
        time = w.time
        for _ in range(12):
            enc = app.device.create_command_encoder()
            app.advance(enc, 1 / 60)
            w.render(enc)
            app.device.submit_command_buffer(enc.finish())
        self.assertFalse(w.follow)
        self.assertGreater(w.domain_center[0], 1)
        self.assertEqual(w.time, time)
        for current, expected in zip([app.position] + [n.position for n in app.npcs], roots):
            np.testing.assert_array_equal(current, expected)
        for foot, expected in zip(w.feet, poses):
            np.testing.assert_array_equal(foot.position, expected)
        app.cut_camera(64)
        enc = app.device.create_command_encoder()
        w.render(enc)
        app.device.submit_command_buffer(enc.finish())
        np.testing.assert_array_equal(w.read_state(), 0)
        self.assertTrue(np.all(w.read_optics()[..., 1] > 0))
        app.device.wait()


if __name__ == "__main__":
    unittest.main()
