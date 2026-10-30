import unittest

import numpy as np

from ..parameters import Fuel
from ..presets import Chaparral, ShortGrass, TimberLitterUnderstory
from ..suppression import (
    DEFAULT_FUEL_GROUP,
    alpha_full_dose,
    cell_alpha,
    depth_needed_mm,
    footprint_line_depths_mm,
    fuel_group,
)

SPRUCE = ("spruce", "conifer_with_litter")


class TestSuppression(unittest.TestCase):
    def test_alpha_full_dose(self) -> None:
        """
        Test the water multiplier 0.38 R^-0.37 at 0.3, 2, and 10 m/min, its cap at 1
        for slow fires, and the constant retardant multiplier
        """
        alpha = alpha_full_dose([0.3, 2.0, 10.0])
        np.testing.assert_allclose(alpha, [0.60, 0.30, 0.16], atol=0.01)
        self.assertEqual(alpha_full_dose(0.01), 1.0)
        self.assertEqual(alpha_full_dose(10.0, agent="retardant"), 0.05)

    def test_depth_needed_mm(self) -> None:
        """
        Test the needed depth on the measured curve and its proportional extension
        below 1,000 kW/m
        """
        need = depth_needed_mm([500.0, 1000.0, 2000.0, 5000.0], "spruce")
        np.testing.assert_allclose(need, [0.425, 0.85, 1.325, 2.65])

    def test_footprint_line_depths_mm(self) -> None:
        """
        Test the depth each pixel of a line 36 m apart gets from one load
        """
        np.testing.assert_allclose(footprint_line_depths_mm("kmax", 36), [1, 2.5, 1])
        np.testing.assert_allclose(
            footprint_line_depths_mm("cl415", 36), [0.25, 1.5, 1.5, 0.25]
        )
        self.assertEqual(len(footprint_line_depths_mm("rj85", 36)), 7)

    def test_cell_alpha(self) -> None:
        """
        Test the worked examples: a K-MAX load acts in full on a cool edge, in part on
        a hot head, and the shallow ends of a CL-415 load act in part
        """
        groups = np.array([SPRUCE] * 3)
        depths = footprint_line_depths_mm("kmax", 36)
        cool = cell_alpha(2.0, 1000.0, depths, groups)
        np.testing.assert_allclose(cool, [0.30, 0.30, 0.30], atol=0.01)
        head = cell_alpha(10.0, 4000.0, depths, groups)
        np.testing.assert_allclose(head, [0.44, 0.16, 0.44], atol=0.01)

        groups = np.array([SPRUCE] * 4)
        depths = footprint_line_depths_mm("cl415", 36)
        ends = cell_alpha(2.0, 1000.0, depths, groups)
        np.testing.assert_allclose(ends, [0.70, 0.30, 0.30, 0.70], atol=0.01)

        groups = np.array([SPRUCE] * 7)
        depths = footprint_line_depths_mm("rj85", 36)
        retardant = cell_alpha(2.0, 1000.0, depths, groups, agent="retardant")
        self.assertAlmostEqual(retardant[0], 0.40, places=2)
        self.assertAlmostEqual(retardant[3], 0.05, places=6)

        # No water, no effect
        none = cell_alpha(2.0, 1000.0, [0.0], np.array([SPRUCE]))
        np.testing.assert_allclose(none, [1.0])

    def test_fuel_group(self) -> None:
        """
        Test that presets map to their groups and other fuels to the default
        """
        self.assertEqual(fuel_group(ShortGrass), ("grass", "grass"))
        self.assertEqual(fuel_group(Chaparral), ("pine", "chaparral_and_heavy_slash"))
        self.assertEqual(fuel_group(TimberLitterUnderstory), SPRUCE)
        self.assertEqual(fuel_group(Fuel(0.1, 1.0, 0.2, 2000)), DEFAULT_FUEL_GROUP)
