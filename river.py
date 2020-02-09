from dataclasses import dataclass
import numpy as np
import opensimplex


@dataclass
class River:
    id: int
    depth: int
    cells: [(int, int)]
    origin_cells: [(int, int)]


@dataclass
class RiverLand:
    river: River
    smooth: float
    cells: [(int, int)]


def norm_noise_func(noise_func):
    return lambda x, y: np.interp(noise_func(x, y), [-1, 1], [0, 1])


relief_noise = norm_noise_func(opensimplex.OpenSimplex(0).noise2d)
river_noise = norm_noise_func(opensimplex.OpenSimplex(1).noise2d)


def sgn(val: int) -> int:
    return 1 if val >= 0 else -1


def gen_cells(x0: int, y0: int, x1: int, y1: int) -> [(int, int)]:
    return [(y, x)
            for x in range(x0, x1 + 1, sgn(x1 - x0))
            for y in range(y0, y1 + 1, sgn(y1 - y0))]


@dataclass
class MainRiverParams:
    width: int
    depth: int
    relief_shape: (int, int)


# Create horizontal main river at center of relief
def create_main_river(params: MainRiverParams) -> River:
    x0 = 0
    x1 = params.relief_shape[1] - 1
    y0 = params.relief_shape[0] // 2 - params.width // 2
    y1 = params.relief_shape[0] // 2 + params.width // 2
    cells = gen_cells(x0, y0, x1, y1)
    origin_cells = gen_cells(0, y0, 0, y1)
    return River(0, params.depth, cells, origin_cells)


@dataclass
class RiverLandsParams:
    main_river: River
    width: int
    depth: int
    count: int
    smooth: float
    relief_shape: (int, int)


# Create vertical riverbeds aligned by sides of main river
def create_river_lands(params: RiverLandsParams) -> [RiverLand]:
    side_river_step = params.relief_shape[1] // (params.count + 1)
    main_river_y0 = min(params.main_river.cells, key=lambda cell: cell[0])[0]
    main_river_y1 = max(params.main_river.cells, key=lambda cell: cell[0])[0]

    river_lands = []

    for i in range(params.count):
        is_even = i % 2 == 0
        river_len = int(params.relief_shape[0] * 0.4)
        river_x0 = side_river_step * (i + 1)
        river_x1 = river_x0 + params.width
        river_y0 = main_river_y0 - 1 if is_even else main_river_y1 + 1
        river_y1 = river_y0 - river_len if is_even else river_y0 + river_len

        river_cells = gen_cells(river_x0, river_y0, river_x1, river_y1)
        river_origin_cells = gen_cells(river_x0, river_y0, river_x1, river_y0)

        side_river = River(i + 1, params.depth, river_cells, river_origin_cells)

        river_land_half_width = params.relief_shape[1] // params.count // 2

        # river_land_len = int(params.relief_shape[0] * 0.45)
        river_land_x0 = river_x0 - river_land_half_width
        river_land_x1 = river_x1 + river_land_half_width
        river_land_y0 = river_y0 - 1 if is_even else river_y0 + 1
        river_land_y1 = 0 if is_even else params.relief_shape[0] - 1
        river_land_cells = gen_cells(river_land_x0, river_land_y0, river_land_x1, river_land_y1)

        river_land = RiverLand(side_river, params.smooth, river_land_cells)
        river_lands.append(river_land)

    return river_lands


def dig_river(
        river: River,
        relief_vals: np.ndarray,
        save_relief: bool = True,
        with_noise: bool = True,
        noise_smooth: int = 0.1
) -> np.ndarray:
    result = relief_vals.copy()
    for cell in river.cells:
        noise_value = min(river_noise(cell[1] * 0.1, cell[0] * noise_smooth) + 0.5, 1) if with_noise else 1
        if save_relief:
            result[cell] += river.depth * noise_value
        else:
            result[cell] = river.depth * noise_value
    return result


