import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def draw_map(values: np.ndarray):
    plt.gcf().set_size_inches(5, 5)
    plt.imshow(values, aspect='equal')
    plt.colorbar()
    plt.show()


def draw_water_distribution(result, count):
    cols = 4
    rows = int(np.ceil(count / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
    for it in range(count):
        x, y = it // 4, it % 4
        axes[x, y].imshow(result.h[it],
                          norm=colors.Normalize(vmin=np.min(result.h[it]), vmax=np.mean(result.h[it])),
                          cmap='Blues')
        axes[x, y].set_title(f't = {it}')

    plt.show()
