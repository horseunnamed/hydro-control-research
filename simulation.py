from dataclasses import dataclass
import numpy as np
import grd
import pathlib
import subprocess
from shutil import copyfile


@dataclass
class Source:
    c1: (int, int)
    c2: (int, int)
    rotate: int
    qs: [float]

    def __hash__(self):
        return hash((
            self.c1,
            self.c2,
            self.rotate,
            tuple(self.qs)
        ))


@dataclass
class Experiment:
    relief: grd.GridMap
    it_secs: int
    days: int
    sources: [Source]

    def __hash__(self):
        return hash((
            self.relief,
            self.it_secs,
            self.days,
            tuple(self.sources)))


@dataclass
class Result:
    h: [np.array]
    v: [(np.array, np.array)]


RELIEF_FNAME = 'relief.grd'
SOURCES_FNAME = 'sources.dat'
HYDROGRAPH_FNAME = 'hydrograph_dts.dat'
SWCUDA_FNAME = 'SW-CUDA_NSource.exe'
CUDALIB_FNAME = 'cudart64_80.dll'
EXPHASH_FNAME = 'hash.dat'


def write(exp: Experiment, dir: str):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    grd.write(exp.relief, dir + '/' + RELIEF_FNAME)
    with open(dir + '/' + HYDROGRAPH_FNAME, 'w') as f:
        np.savetxt(f, [exp.it_secs, 1, len(exp.sources), exp.days], fmt='%g')
        for source in exp.sources:
            np.savetxt(f, source.qs, fmt='%g')
    with open(dir + '/' + SOURCES_FNAME, 'w') as f:
        for source in exp.sources:
            np.savetxt(
                f,
                [source.rotate, source.c1[0], source.c2[0], source.c1[1], source.c2[1]],
                fmt='%g')


def read(dir: str) -> Experiment:
    relief = grd.read(dir + '/' + RELIEF_FNAME)
    with open(dir + '/' + HYDROGRAPH_FNAME, 'r') as f:
        it_secs = int(f.readline())
        velocity = float(f.readline())
        sources_num = int(f.readline())
        days = int(f.readline())
        qs = [np.loadtxt(f, delimiter='\n', max_rows=days) for _ in range(sources_num)]
    with open(dir + '/' + SOURCES_FNAME, 'r') as f:
        sources = []
        for i in range(sources_num):
            rotate = float(f.readline())
            x1 = int(f.readline())
            x2 = int(f.readline())
            y1 = int(f.readline())
            y2 = int(f.readline())
            sources.append(Source((x1, y1), (x2, y2), rotate, qs[i]))
    return Experiment(relief, it_secs, days, sources)


def get_fname_for(prefix: str, it: int) -> str:
    it_str = (' ' if it < 10 else '') + str(it)
    return f'{prefix}_   {it_str}.grd'


def read_result(exp: Experiment, dir: str) -> Result:
    result = Result([], [])
    for it in range((exp.days - 1) * 24):
        h = grd.read(dir + '/' + get_fname_for('H', it))
        vx = grd.read(dir + '/' + get_fname_for('vx', it))
        vy = grd.read(dir + '/' + get_fname_for('vy', it))
        result.h.append(h.vals)
        result.v.append((vx.vals, vy.vals))
    return result


def run(exp: Experiment, dir: str, sw_cuda_dir: str = 'sw-cuda') -> Result:
    write(exp, dir)

    def copy_if_not_exists(fname: str):
        if not pathlib.Path(dir + '/' + fname).exists():
            copyfile(sw_cuda_dir + '/' + fname, dir + '/' + fname)

    copy_if_not_exists(SWCUDA_FNAME)
    copy_if_not_exists(CUDALIB_FNAME)

    hash_file_path = dir + '/' + EXPHASH_FNAME
    exp_hash = str(hash(exp))
    if pathlib.Path(hash_file_path).exists():
        with open(hash_file_path, 'r') as f:
            given_hash = f.readline()
        if exp_hash == given_hash:
            return read_result(exp, dir)

    command = f'start cmd /k "cd {dir} && (echo 0.06 | {SWCUDA_FNAME}) && exit"'
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE, shell=True, stderr=subprocess.STDOUT, bufsize=1, close_fds=True)
    for line in iter(process.stdout.readline, b''):
        print(line)
    process.stdout.close()
    process.wait()

    with open(hash_file_path, 'w') as f:
        f.write(exp_hash)

    return read_result(exp, dir)


def q_at(result: Result, cells: [(int, int)], it: int) -> float:
    def f(cell: (int, int)):
        x, y = cell
        return result.h[it][x, y] * (result.v[it][0][x, y] ** 2 + result.v[it][1][x, y])

    return sum(map(f, cells))
