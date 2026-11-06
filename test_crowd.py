"""Multi-character movement and real GPU interaction/visibility regressions."""
import unittest

import numpy as np

from app import parse_args
from crowd import Wanderer
from gait import Biped, water_margin
from water import Settings, Water, create_device


class CrowdChecks(unittest.TestCase):
    def test_defaults_and_seeded_safe_walks(self):
        args = parse_args([])
        self.assertEqual((args.zoom, args.damping, args.npcs), (1.4, 1.3, 3))
        for index in range(3):
            npc = Wanderer(index)
            runs = []
            for _ in range(2):
                npc.reset(0.12, False)
                samples = []
                for frame in range(1800):
                    previous = npc.position.copy()
                    feet = npc.update(1 / 60, 0.12, False)
                    self.assertLessEqual(np.linalg.norm(npc.position - previous), 1.45 / 60 + 1e-8)
                    self.assertGreater(water_margin(npc.position), 0.9)
                    for foot in feet:
                        self.assertGreater(water_margin(foot.position[[0, 2]]), 0.1)
                    if frame % 60 == 0:
                        samples.append(npc.position.copy())
                self.assertGreater(min(npc.gait.landings), 12)
                runs.append(samples)
            np.testing.assert_array_equal(runs[0], runs[1])
            self.assertGreater(np.ptp(runs[0], axis=0).max(), 2)

    def test_all_eight_feet_force_water_and_are_refracted(self):
        device = create_device()
        w = Water(device, Settings(resolution=256, open_water=True, water_depth=0.25), 640, 480)
        w.feet = []
        roots = [(-2, -2), (2, -2), (-2, 2), (2, 2)]
        for root in roots:
            gait = Biped()
            gait.reset(root, depth=0.25, open_water=True)
            for foot in gait.feet:
                foot.previous[1] = 0.3
            w.feet.extend(gait.feet)
        w.view, w.top_down, w.detail, w.normal_strength = 4, True, 0, 0
        encoder = device.create_command_encoder()
        w.step(encoder)
        w.render(encoder)
        device.submit_command_buffer(encoder.finish())
        state, optics = w.read_state(), w.read_optics()
        for foot, material in zip(w.feet, (1, 2, 5, 6, 7, 8, 9, 10)):
            cell = ((foot.position[[0, 2]] / 24 + 0.5) * 256).astype(int)
            x, z = cell
            self.assertGreater(np.max(np.abs(state[z-1:z+2, x-1:x+2, 1])), 0.01)
            self.assertGreater(np.count_nonzero(optics[..., 2] == material), 2)
        self.assertTrue(np.isfinite(state).all())
        device.wait()


if __name__ == "__main__":
    unittest.main()
