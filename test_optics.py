"""Real GPU optical invariants and world-space visibility regression tests."""
import math
import unittest

import numpy as np
import slangpy as spy

from gait import bottom_height
from water import Settings, Water, create_device


class OpticalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device()
        cls.water = Water(cls.device, Settings(resolution=64, open_water=True), 513, 385)

    @classmethod
    def tearDownClass(cls):
        cls.device.wait()

    def setUp(self):
        w = self.water
        w.settings = Settings(resolution=64, open_water=True)
        w.player = (0, 0)
        w.detail, w.normal_strength = 0.0, 0.0
        w.zoom, w.follow, w.foot_debug = 5.0, True, False
        w.top_down, w.view = False, 0
        w.ior, w.roughness, w.absorption = 1.333, 0.12, 1.0
        w.refraction, w.reflection, w.floor_grid = True, True, False
        w.reset()

    def draw(self):
        encoder = self.device.create_command_encoder()
        self.water.render(encoder)
        self.device.submit_command_buffer(encoder.finish())
        optics = self.water.read_optics()
        image = self.water.display.to_numpy().copy()
        self.assertTrue(np.isfinite(optics).all())
        self.assertTrue(np.isfinite(image).all())
        self.assertGreaterEqual(optics[..., :2].min(), 0)
        self.assertLessEqual(optics[..., 0].max(), 1)
        return image, optics

    def test_gpu_snell_fresnel_beer_and_intersections(self):
        texture = self.device.create_texture(width=6, height=1, format=spy.Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)
        kernel = self.device.create_compute_kernel(self.device.load_program("test_optics.slang", ["probe"]))
        kernel.dispatch(thread_count=[6, 1, 1], vars={"g_results": texture})
        self.device.wait()
        result = texture.to_numpy()[0]
        self.assertTrue(np.isfinite(result).all())
        f0 = ((1.333 - 1) / (1.333 + 1))**2
        np.testing.assert_allclose(result[0], [f0, 1, 0, 1], atol=1e-6)
        sin_t = 0.6 / 1.333
        np.testing.assert_allclose(result[1, :3], [sin_t, -math.sqrt(1 - sin_t**2), 0], atol=1e-6)
        np.testing.assert_allclose(result[2], [1.5, 0.5, -1, 2], atol=1e-6)
        np.testing.assert_allclose(result[3, :3], np.exp(-np.array([0.2, 1, 2]) * 0.3), atol=1e-6)
        np.testing.assert_allclose(result[4, :3], result[3, :3]**2, atol=1e-6)
        self.assertGreater(result[5, 0], 0)
        np.testing.assert_array_equal(result[5, 1:], 0)

    def test_flat_water_path_matches_snell_and_shortens_vs_straight_ray(self):
        _, optics = self.draw()
        sin_i = 0.78 / math.sqrt(1 + 0.78**2)
        expected = 0.12 / math.sqrt(1 - (sin_i / 1.333)**2)
        floor = (optics[..., 1] > 0) & (optics[..., 2] == 0)
        np.testing.assert_allclose(optics[..., 1][floor], expected, atol=2e-5)
        self.water.refraction = False
        _, straight = self.draw()
        floor = (straight[..., 1] > 0) & (straight[..., 2] == 0)
        np.testing.assert_allclose(straight[..., 1][floor], 0.12 * math.sqrt(1 + 0.78**2), atol=2e-5)
        self.assertGreater(straight[..., 1].max(), expected + 0.01)

    def test_submerged_foot_silhouette_moves_with_refraction(self):
        self.water.settings.water_depth = 0.25
        self.water.reset()  # Both feet entirely below the interface.
        self.water.view = 4
        bent_image, bent = self.draw()
        self.water.refraction = False
        straight_image, straight = self.draw()
        for foot_id in (1, 2):
            bent_y = np.nonzero(bent[..., 2] == foot_id)[0]
            straight_y = np.nonzero(straight[..., 2] == foot_id)[0]
            self.assertGreater(len(bent_y), 100)
            self.assertGreater(len(straight_y), 100)
            self.assertGreater(abs(bent_y.mean() - straight_y.mean()), 1.0)
        self.assertGreater(np.abs(bent_image - straight_image).max(), 0.1)

    def test_reflection_sees_a_foot_outside_primary_view(self):
        # At this view's upper edge the primary foot projection is out of frame,
        # but its mirror image lies inside. SSR cannot recover this geometry.
        self.water.feet[0].position[:] = [0, 0.8, -3.0]
        self.water.feet[1].position[:] = [30, 0.8, 30]
        self.water.view = 5
        sharp, optics = self.draw()
        self.assertGreater(np.count_nonzero(optics[..., 3] == 1), 100)
        self.assertTrue(np.all(optics[..., 1] > 0))  # No foreground occlusion.
        self.assertFalse(np.any(optics[..., 2] == 1))  # No submerged ghost.
        self.water.roughness = 0.4
        rough, _ = self.draw()
        self.assertGreater(np.abs(sharp - rough).max(), 0.1)
        self.water.reflection = False
        disabled, data = self.draw()
        np.testing.assert_array_equal(data[..., 3], 0)
        np.testing.assert_array_equal(disabled[..., :3], 0)

    def test_equal_ior_eliminates_bending_and_fresnel(self):
        self.water.ior = 1.0
        self.water.floor_grid = True
        image, optics = self.draw()
        np.testing.assert_array_equal(optics[..., 0], 0)
        self.water.refraction = False
        self.water.reflection = False
        straight, _ = self.draw()
        np.testing.assert_allclose(image, straight, atol=2e-5)

    def test_world_space_bank_intersection_and_zero_depth(self):
        self.water.settings.open_water = False
        self.water.settings.water_depth = 0.25
        self.water.player = (9.7, 0)
        self.water.reset()
        _, data = self.draw()
        sin_t = (0.78 / math.sqrt(1 + 0.78**2)) / self.water.ior
        direction = np.array([0, -math.sqrt(1 - sin_t**2), -sin_t])
        bank_samples = 0
        for y in range(0, 385, 12):
            for x in range(0, 513, 12):
                f, path, transmitted, _ = data[y, x]
                if f <= 0 or transmitted != 0:
                    continue
                p = self.water.screen_to_world(x + 0.5, y + 0.5)
                hit = np.array([p[0], 0, p[1]]) + direction * path
                floor = bottom_height(hit[[0, 2]], 0.25)
                self.assertAlmostEqual(hit[1], floor, delta=0.0004)
                bank_samples += -0.24 < floor < -0.005
        self.assertGreater(bank_samples, 10)
        self.water.settings.water_depth = 0
        _, data = self.draw()
        np.testing.assert_array_equal(data, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
