from dataclasses import dataclass
import numpy as np


@dataclass
class GridMap:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    vals: np.ndarray

    def size_x(self) -> int:
        return self.vals.shape[0]

    def size_y(self) -> int:
        return self.vals.shape[1]

    def min_z(self) -> float:
        return np.min(self.vals)

    def max_z(self) -> float:
        return np.max(self.vals)

    def __hash__(self):
        return hash((
            self.min_x,
            self.max_x,
            self.min_y,
            self.max_y,
            self.vals.data.tobytes()))


def create_map(vals: np.ndarray, cell_size: int) -> GridMap:
    return GridMap(0, (vals.shape[1] - 1) * cell_size, 0, (vals.shape[0] - 1) * cell_size, vals)


def read(grd_fname) -> GridMap:
    with open(grd_fname, 'rb') as f:
        np.fromfile(f, 'b', 4)
        size_x, size_y = np.fromfile(f, np.int16, 2)
        min_x, max_x, min_y, max_y = np.fromfile(f, np.double, 4)
        np.fromfile(f, np.double, 2)
        vals = np.fromfile(f, np.float32)
        return GridMap(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            vals=np.reshape(vals, (size_x, size_y))
        )


def write(grid_map, grd_fname):
    with open(grd_fname, 'wb') as f:
        np.array(b'DSBB').tofile(f)
        np.array([grid_map.size_x(), grid_map.size_y()], dtype=np.int16).tofile(f)
        np.array([
            grid_map.min_x, grid_map.max_x,
            grid_map.min_y, grid_map.max_y,
            grid_map.min_z(), grid_map.max_z()], dtype=np.float64).tofile(f)
        grid_map.vals.astype(np.float32).tofile(f)
