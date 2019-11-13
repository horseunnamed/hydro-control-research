from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass
class GridMap:
    minX: float
    maxX: float
    minY: float
    maxY: float
    vals: np.array

    def sizeX(self) -> int:
        return self.vals.shape[0]

    def sizeY(self) -> int:
        return self.vals.shape[1]

    def minZ(self) -> float:
        return np.min(self.vals)

    def maxZ(self) -> float:
        return np.max(self.vals)

    def __hash__(self):
        return hash((
            self.minX,
            self.maxX,
            self.minY,
            self.maxY,
            self.vals.data.tobytes()))

def createMap(sizeX, sizeY, cellSize, val) -> GridMap:
    return GridMap(0, (sizeX - 1) * cellSize, 0, (sizeY - 1) * cellSize,
        np.full((sizeX, sizeY), val))

def read(grdFname) -> GridMap:
    with open(grdFname, 'rb') as f:
        np.fromfile(f, 'b', 4)
        sizeX, sizeY = np.fromfile(f, np.int16, 2)
        minX, maxX, minY, maxY = np.fromfile(f, np.double, 4)
        np.fromfile(f, np.double, 2)
        vals = np.fromfile(f, np.float32)
        return GridMap(
            minX = minX,
            maxX = maxX,
            minY = minY,
            maxY = maxY,
            vals = np.reshape(vals, (sizeX, sizeY))
        )

def write(gridMap, grdFname):
    with open(grdFname, 'wb') as f:
        np.array(b'DSBB').tofile(f)
        np.array([gridMap.sizeX(), gridMap.sizeY()], dtype=np.int16).tofile(f)
        np.array([
            gridMap.minX, gridMap.maxX,
            gridMap.minY, gridMap.maxY, 
            gridMap.minZ(), gridMap.maxZ()], dtype=np.float64).tofile(f)
        gridMap.vals.astype(np.float32).tofile(f)