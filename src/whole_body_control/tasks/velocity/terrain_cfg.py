"""Terrain generators for VITA BOY locomotion tasks."""

from dataclasses import dataclass

import mjlab.terrains as terrain_gen
import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border, make_plane


@dataclass(kw_only=True)
class BoxPitTerrainCfg(SubTerrainCfg):
    """A stepped pit with an optional second level."""

    pit_depth_range: tuple[float, float]
    platform_width: float = 1.0
    double_pit: bool = False

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        pit_depth = self.pit_depth_range[0] + difficulty * (self.pit_depth_range[1] - self.pit_depth_range[0])
        total_depth = pit_depth * (2.0 if self.double_pit else 1.0)
        body = spec.body("terrain")

        if total_depth <= 1.0e-6:
            ground = make_plane(body, self.size, 0.0, center_zero=False)
            return TerrainOutput(
                origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]),
                geometries=[TerrainGeometry(geom=geom) for geom in ground],
            )

        inner_size = (self.platform_width, self.platform_width)
        geometries = []
        if self.double_pit:
            ratio = 0.6
            inner_size = (
                self.platform_width + (self.size[0] - self.platform_width) * ratio,
                self.platform_width + (self.size[1] - self.platform_width) * ratio,
            )

        center = (self.size[0] / 2, self.size[1] / 2, -total_depth / 2)
        geometries.extend(make_border(body, self.size, inner_size, total_depth, center))

        if self.double_pit:
            inner_center = (self.size[0] / 2, self.size[1] / 2, -total_depth)
            geometries.extend(
                make_border(
                    body,
                    inner_size,
                    (self.platform_width, self.platform_width),
                    total_depth,
                    inner_center,
                )
            )

        geometries.extend(make_plane(body, self.size, -total_depth, center_zero=False))
        return TerrainOutput(
            origin=np.array([self.size[0] / 2, self.size[1] / 2, -total_depth]),
            geometries=[TerrainGeometry(geom=geom) for geom in geometries],
        )


GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.02, 0.04),
            noise_step=0.02,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            border_width=0.25,
        ),
    },
    add_lights=True,
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "pyramid_stairs_28": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.0, 0.15),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(-0.02, 0.04),
            noise_step=0.02,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            border_width=0.25,
            scale_with_difficulty=True,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.0, 0.2),
            num_waves=5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
        ),
        "high_platform": BoxPitTerrainCfg(
            proportion=0.15,
            pit_depth_range=(0.0, 0.3),
            platform_width=2.0,
            double_pit=True,
        ),
    },
    add_lights=True,
)
