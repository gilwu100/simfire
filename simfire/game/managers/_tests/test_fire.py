import unittest

import numpy as np

from ....enums import BurnStatus, GameStatus
from ....game._tests import DummyFuelLayer, DummyTopographyLayer
from ....game.managers.mitigation import FireLineManager
from ....utils.config import Config
from ....world.parameters import Environment, FuelParticle
from ....world.suppression import BTU_PER_FT_S_TO_KW_PER_M, FT_PER_MIN_TO_M_PER_MIN
from ...sprites import Fire, Terrain
from ..fire import ConstantSpreadFireManager, FireManager, RothermelFireManager


class TestFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.init_pos = (
            self.config.area.screen_size[0] // 2,
            self.config.area.screen_size[1] // 2,
        )
        self.fire_size = self.config.display.fire_size
        self.max_fire_duration = self.config.fire.max_fire_duration

        self.fire_manager = FireManager(
            self.init_pos, self.fire_size, self.max_fire_duration
        )

    def test_update(self) -> None:
        """Test that the call to update() just passes"""
        self.assertIsNone(
            self.fire_manager.update(np.array([[1]])),
            msg="The FireManager.update() method should return None",
        )

    def test_prune_sprites(self) -> None:
        """
        Test that the sprites are pruned correctly.
        """
        fire_map = np.full(
            self.config.area.screen_size,
            BurnStatus.UNBURNED,
        )
        # Create a sprite that is past the duration
        new_sprite = Fire(self.init_pos, self.fire_size)
        sprites = [new_sprite]
        durations = [self.max_fire_duration + 1]

        self.fire_manager.sprites = sprites
        self.fire_manager.durations = durations
        self.fire_manager._prune_sprites(fire_map)

        self.assertEqual(
            len(self.fire_manager.sprites),
            0,
            msg=(
                "The fire manager did not prune the Fire sprite "
                "with a duration greater than the max duration"
            ),
        )

    def test_get_new_locs(self) -> None:
        """
        Test that new locations can be retrieved correctly.
        """
        fire_map = np.full(
            self.config.area.screen_size,
            BurnStatus.UNBURNED,
        )
        # Use a location that is too large and out-of-bounds
        x, y = self.config.area.screen_size
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        valid_locs = ((x - 1, y - 1),)
        self.assertTupleEqual(
            new_locs,
            valid_locs,
            msg=(
                f"The new locations: {new_locs} do not match "
                f"the valid locations: {valid_locs} when the "
                "location is too large"
            ),
        )

        # Use a location that is too small and out-of-bounds
        x, y = (0, 0)
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        valid_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1))
        self.assertTupleEqual(
            new_locs,
            valid_locs,
            msg=(
                f"The new locations: {new_locs} do not match "
                f"the valid locations: {valid_locs} when the "
                "location is too small"
            ),
        )

        # Use a location that is BURNED
        x, y = (
            self.config.area.screen_size[0] // 2,
            self.config.area.screen_size[1] // 2,
        )
        fire_map[y, x + 1] = BurnStatus.BURNED
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        # All 8-connected points except (x+1, y)
        valid_locs = (
            (x + 1, y + 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x - 1, y),
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
        )
        self.assertTupleEqual(
            new_locs,
            valid_locs,
            msg=(
                f"The new locations: {new_locs} do not match "
                f"the valid locations: {valid_locs} when a "
                "new location is BURNED"
            ),
        )


class TestRothermelFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            "./simfire/utils/_tests/test_configs/test_config_rothermel_manager.yml"
        )
        self.screen_size = self.config.area.screen_size
        self.headless = False

        self.fuel_particle = FuelParticle()
        topo_layer = DummyTopographyLayer(self.screen_size)
        fuel_layer = DummyFuelLayer(self.screen_size)
        self.terrain = Terrain(
            fuel_layer, topo_layer, self.screen_size, headless=self.headless
        )
        # Use simple/constant wind speed
        self.environment = Environment(
            self.config.environment.moisture,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        self.fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
            headless=self.headless,
        )
        self.fire_init_pos = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        self.fire_manager = RothermelFireManager(
            self.fire_init_pos,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            self.fuel_particle,
            self.terrain,
            self.environment,
            max_time=self.config.simulation.runtime,
            headless=self.headless,
        )

    def test_wind_conversion(self) -> None:
        """
        Test that wind speed/directions of type float, nested sequence and numpy array
        are correctly converted to a numpy array.
        """
        # Test for all possible input types
        u_float = 7.0
        u_dir_float = 90.0
        u_seq = [
            [7.0 for x in range(self.screen_size[1])] for y in range(self.screen_size[0])
        ]
        u_dir_seq = [
            [90.0 for x in range(self.screen_size[1])] for y in range(self.screen_size[0])
        ]
        u_arr = np.full(self.screen_size, 7.0)
        u_dir_arr = np.full(self.screen_size, 90.0)
        U = [u_float, u_seq, u_arr]
        U_dir = [u_dir_float, u_dir_seq, u_dir_arr]
        types = ["float", "nested sequence", "numpy array"]
        for u, u_dir, t in zip(U, U_dir, types):
            with self.subTest(f"Checking for wind type {t}"):
                environment = Environment(self.config.environment.moisture, u, u_dir)
                fire_manager = RothermelFireManager(
                    self.fire_init_pos,
                    self.config.display.fire_size,
                    self.config.fire.max_fire_duration,
                    self.config.area.pixel_scale,
                    self.config.simulation.update_rate,
                    self.fuel_particle,
                    self.terrain,
                    environment,
                    max_time=self.config.simulation.runtime,
                    headless=self.headless,
                )
                self.assertIsInstance(
                    fire_manager.U,
                    np.ndarray,
                    msg="The wind speed in the fire manager should be "
                    "a numpy array, but got "
                    f"{type(fire_manager.U)}",
                )
                self.assertIsInstance(
                    fire_manager.U_dir,
                    np.ndarray,
                    msg="The wind direction in the fire manager should "
                    "be a numpy array, but got "
                    f"{type(fire_manager.U)}",
                )
                arr_shape = fire_manager.U.shape
                self.assertTupleEqual(
                    arr_shape,
                    self.screen_size,
                    msg="The wind speed array should have shape "
                    f"{self.screen_size}, but got {arr_shape}",
                )
                arr_shape = fire_manager.U_dir.shape
                self.assertTupleEqual(
                    arr_shape,
                    self.screen_size,
                    msg="The wind direction array should have shape "
                    f"{self.screen_size}, but got {arr_shape}",
                )

        # Test for exceptions
        with self.assertRaises(
            ValueError, msg="Numpy arr with wrong shape raises ValueError"
        ):
            wrong_shape = (self.screen_size[0] + 1, self.screen_size[1] + 1)
            u_arr = np.full(wrong_shape, 7.0)
            u_dir_arr = np.full(wrong_shape, 90.0)
            environment = Environment(self.config.environment.moisture, u_arr, u_dir_arr)
            fire_manager = RothermelFireManager(
                self.fire_init_pos,
                self.config.display.fire_size,
                self.config.fire.max_fire_duration,
                self.config.area.pixel_scale,
                self.config.simulation.update_rate,
                self.fuel_particle,
                self.terrain,
                environment,
                max_time=self.config.simulation.runtime,
                headless=self.headless,
            )
        with self.assertRaises(
            ValueError, msg="Nested sequence with wrong shape raises ValueError"
        ):
            wrong_shape = (self.screen_size[0] + 1, self.screen_size[1] + 1)
            u_seq = [[7.0 for x in range(wrong_shape[1])] for y in range(wrong_shape[0])]
            u_dir_seq = [
                [90.0 for x in range(wrong_shape[1])] for y in range(wrong_shape[0])
            ]
            environment = Environment(self.config.environment.moisture, u_seq, u_dir_seq)
            fire_manager = RothermelFireManager(
                self.fire_init_pos,
                self.config.display.fire_size,
                self.config.fire.max_fire_duration,
                self.config.area.pixel_scale,
                self.config.simulation.update_rate,
                self.fuel_particle,
                self.terrain,
                environment,
                max_time=self.config.simulation.runtime,
                headless=self.headless,
            )
        with self.assertRaises(ValueError, msg="Non-nested sequence raises ValueError"):
            u_seq = [7.0 for x in range(wrong_shape[1])]
            u_dir_seq = [90.0 for x in range(wrong_shape[1])]
            environment = Environment(self.config.environment.moisture, u_seq, u_dir_seq)
            fire_manager = RothermelFireManager(
                self.fire_init_pos,
                self.config.display.fire_size,
                self.config.fire.max_fire_duration,
                self.config.area.pixel_scale,
                self.config.simulation.update_rate,
                self.fuel_particle,
                self.terrain,
                environment,
                max_time=self.config.simulation.runtime,
                headless=self.headless,
            )

    def test_update_rate_of_spread(self) -> None:
        """
        Test updating the rate of spread based on locations of control lines
        """
        fire_map = np.full(
            self.config.area.screen_size,
            BurnStatus.UNBURNED,
        )
        rate_of_spread = np.zeros_like(fire_map)

        fireline_y_coords = [0, 0]
        fireline_x_coords = [0, 1]
        scratchline_y_coords = [0, 0]
        scratchline_x_coords = [2, 3]
        wetline_y_coords = [0, 0]
        wetline_x_coords = [4, 5]

        y_coords = [0, 0, 0, 0, 0, 0]
        x_coords = [0, 1, 2, 3, 4, 5]

        fire_map[fireline_y_coords, fireline_x_coords] = BurnStatus.FIRELINE
        fire_map[scratchline_y_coords, scratchline_x_coords] = BurnStatus.SCRATCHLINE
        fire_map[wetline_y_coords, wetline_x_coords] = BurnStatus.WETLINE
        rate_of_spread[y_coords, x_coords] = 1

        # First, test the True case
        self.fire_manager.attenuate_line_ros = True
        rate_of_spread_true = self.fire_manager._update_rate_of_spread(
            rate_of_spread, fire_map
        )
        # Check to make sure that the rate of spread was changed to a smaller value
        self.assertLess(np.sum(rate_of_spread_true), 5)

        # Then, test the False case
        self.fire_manager.attenuate_line_ros = False
        rate_of_spread_false = self.fire_manager._update_rate_of_spread(
            rate_of_spread, fire_map
        )
        self.assertEqual(np.sum(rate_of_spread_false), 0)

    def test_direct_attack_default_alpha(self) -> None:
        """
        Test that a direct attack given neither alpha nor depth applies
        `direct_attack_alpha` to its pixels until it expires, and that a second one
        on the same pixel extends the effect without compounding it
        """
        self.fire_manager.direct_attack_alpha = 0.5
        self.fire_manager.activate_direct_attack([(1, 2), (3, 4)], duration=20)
        self.fire_manager.activate_direct_attack([(1, 2)], duration=30)

        multiplier = self.fire_manager.get_direct_attack_multiplier()
        self.assertEqual(multiplier[2, 1], 0.5)
        self.assertEqual(multiplier[4, 3], 0.5)
        self.assertEqual(np.sum(multiplier != 1), 2)
        np.testing.assert_array_equal(
            self.fire_manager.get_direct_attack_mask(), multiplier != 1
        )

        self.fire_manager.elapsed_time += 20
        multiplier = self.fire_manager.get_direct_attack_multiplier()
        self.assertEqual(multiplier[2, 1], 0.5)
        self.assertEqual(multiplier[4, 3], 1)

        self.fire_manager.elapsed_time += 10
        multiplier = self.fire_manager.get_direct_attack_multiplier()
        self.assertTrue(np.all(multiplier == 1))

    def test_direct_attack_per_pixel_alpha(self) -> None:
        """
        Test that each pixel keeps its own alpha, that overlapping direct attacks
        multiply, and that the overlap ends when the shorter one expires
        """
        self.fire_manager.activate_direct_attack(
            [(1, 2), (3, 4)], duration=30, alpha=[0.2, 0.6]
        )
        self.fire_manager.activate_direct_attack([(1, 2)], duration=10, alpha=0.5)

        multiplier = self.fire_manager.get_direct_attack_multiplier()
        self.assertAlmostEqual(multiplier[2, 1], 0.1, places=6)
        self.assertAlmostEqual(multiplier[4, 3], 0.6, places=6)

        self.fire_manager.elapsed_time += 10
        multiplier = self.fire_manager.get_direct_attack_multiplier()
        self.assertAlmostEqual(multiplier[2, 1], 0.2, places=6)
        self.assertEqual(len(self.fire_manager.direct_attacks), 1)

        with self.assertRaises(ValueError):
            self.fire_manager.activate_direct_attack([(1, 2)], alpha=1.5)
        with self.assertRaises(ValueError):
            self.fire_manager.activate_direct_attack([(1, 2)], alpha=0.5, depth_mm=1)

    def test_direct_attack_depth(self) -> None:
        """
        Test that a direct attack given as a depth takes its alpha from the spread
        rate, intensity, and fuel of each pixel (all Chaparral here, which needs
        0.75 mm of water at 1,000 kW/m)
        """
        points = [(1, 2), (3, 4)]
        self.fire_manager.activate_direct_attack(points, depth_mm=[1.0, 0.375])
        rows, cols = [2, 4], [1, 3]
        # 2 m/min and 1,000 kW/m in the manager's units
        self.fire_manager.incoming_ros[rows, cols] = 2 / FT_PER_MIN_TO_M_PER_MIN
        self.fire_manager.incoming_intensity[rows, cols] = 1000 / BTU_PER_FT_S_TO_KW_PER_M

        multiplier = self.fire_manager.get_direct_attack_multiplier()
        full = 0.38 * 2**-0.37
        # 1.0 mm is more than the need, 0.375 mm is half of it
        self.assertAlmostEqual(multiplier[2, 1], full, places=2)
        self.assertAlmostEqual(multiplier[4, 3], full**0.5, places=2)

        # Where the fire is not spreading in, the multiplier is 1
        self.fire_manager.incoming_ros[rows, cols] = 0
        multiplier = self.fire_manager.get_direct_attack_multiplier()
        self.assertTrue(np.all(multiplier == 1))

    def test_direct_attack_stops_spread(self) -> None:
        """
        Test that pixels under a direct attack with alpha 0 do not ignite while every
        other neighbour of the initial fire does
        """
        fire_map = np.full(self.screen_size, BurnStatus.UNBURNED)
        self.fire_manager.pixel_scale = 0
        x, y = self.fire_init_pos
        new_locs = self.fire_manager._get_new_locs(x, y, fire_map)
        held = list(new_locs[:3])
        self.fire_manager.activate_direct_attack(held, duration=60, alpha=0)

        fire_map, _ = self.fire_manager.update(fire_map)
        for col, row in new_locs:
            with self.subTest(loc=(col, row)):
                if (col, row) in held:
                    self.assertEqual(fire_map[row, col], BurnStatus.UNBURNED)
                else:
                    self.assertEqual(fire_map[row, col], BurnStatus.BURNING)

    def test_update(self) -> None:
        """
        Test that the RothermelFireManager will update correctly. There is no need to
        check the rate of spread calculation since that has its own unit test. Instead,
        check that the fire will spread correctly once enough time has passed.
        """
        # Create simulation parameters that will guarantee fire spread
        fire_map = np.full_like(self.terrain.fuels, BurnStatus.UNBURNED)
        self.fire_manager.pixel_scale = 0
        new_locs = self.fire_manager._get_new_locs(
            self.fire_init_pos[0], self.fire_init_pos[1], fire_map
        )
        new_locs_uzip = tuple(zip(*new_locs))
        self.fire_manager.burn_amounts[new_locs_uzip] = -1

        # Update the manager and get the locations that are now burning
        # These should match new_locs since those locations are guaranteed
        # to burn with a pixel_scale of -1
        fire_map, status = self.fire_manager.update(fire_map)
        burning_locs = np.where(
            self.fire_manager.burn_amounts > self.fire_manager.pixel_scale
        )
        burning_locs = tuple(map(tuple, burning_locs[::-1]))
        burning_locs = tuple(zip(*burning_locs))

        self.assertEqual(
            status, GameStatus.RUNNING, msg=('The game status should be "RUNNING"')
        )

        self.assertCountEqual(
            burning_locs,
            new_locs,
            msg=(
                "The locations that are now burning are: "
                f"{burning_locs}, but they should be: {new_locs}"
            ),
        )

        # Test that the RothermelFireManager sucessfully created the new Fire sprites
        # The 0th entry is the initial fire, so don't include it
        new_sprites = self.fire_manager.sprites[1:]
        num_new_sprites = len(new_sprites)
        num_new_locs = len(new_locs)

        self.assertEqual(
            num_new_sprites,
            num_new_locs,
            msg=(
                "The RothermelFireManager should have created "
                f"{num_new_locs} Fires, but only {num_new_sprites} were "
                "created"
            ),
        )

        # Test that the RothermelFireManager created the sprites in the correct locations
        for i, sprite in enumerate(new_sprites):
            with self.subTest(i=i):
                x, y, _, _ = sprite.rect
                self.assertIn(
                    (x, y),
                    new_locs,
                    msg=(
                        f"A sprite was created at (x, y) location ({x}, {y}), "
                        "but that location is not valid for initital "
                        f"location ({x}, {y}) and new locations {new_locs}"
                    ),
                )

        # Output the draw graph for QA
        fig = self.fire_manager.draw_spread_graph()
        fig.savefig("./assets/fire_spread_graph.png")


class TestConstantSpreadFireManager(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.init_pos = (
            self.config.area.screen_size[0] // 5,
            self.config.area.screen_size[1] // 7,
        )
        self.fire_size = self.config.display.fire_size
        self.max_fire_duration = self.config.fire.max_fire_duration
        self.rate_of_spread = self.max_fire_duration - 1

        self.fire_manager = ConstantSpreadFireManager(
            self.init_pos, self.fire_size, self.max_fire_duration, self.rate_of_spread
        )

    def test_update(self) -> None:
        """
        Test that the ConstantSpreadFireManager will update correctly. This will
        make sure that no fires are spread before max_fire_duration updates, and that the
        correct number of new fires are created at the correct locations.
        """
        fire_map = np.zeros(self.config.area.screen_size)
        # Get the locations where Fires should be created
        new_locs = self.fire_manager._get_new_locs(
            self.init_pos[0], self.init_pos[1], fire_map
        )

        # Update the simulation until 1 update before any spreading
        for i in range(self.rate_of_spread):
            fire_map = self.fire_manager.update(fire_map)
            # Call the fire sprite update here since it's normally handled by PyGame
            # This is needeed to increment the duration
            self.fire_manager.sprites[0].update()
            with self.subTest(update=i):
                # Verify that only the initial Fire sprite exists
                self.assertEqual(len(self.fire_manager.sprites), 1)

        # Update to make the ConstantSpreadFireManager create new Fire sprites
        self.fire_manager.update(fire_map)

        # The 0th entry is the initial fire, so don't include it
        new_sprites = self.fire_manager.sprites[1:]

        # Verify that the correct number of sprites were created
        num_new_sprites = len(new_sprites)
        num_new_locs = len(new_locs)
        self.assertEqual(
            num_new_sprites,
            num_new_locs,
            msg=(
                "The ConstantSpreadFireManager should have created "
                f"{num_new_locs} Fires, but {num_new_sprites} were "
                "created"
            ),
        )

        sprite_locs = tuple(tuple(s.rect[:2]) for s in new_sprites)
        # Verify that the locations of the new Fire sprites are correct
        self.assertCountEqual(
            sprite_locs,
            new_locs,
            msg=(
                "The locations that are now burning are: "
                f"{sprite_locs}, but they should be: {new_locs}"
            ),
        )
