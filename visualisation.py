import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def draw_water_distribution(result, count):
    cols = 4
    rows = int(np.ceil(count / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * rows, 3 * cols))
    for it in range(count):
        x, y = it // 4, it % 4
        axes[x, y].imshow(result.h[it],
                          norm=colors.Normalize(vmin=0, vmax=0.1),
                          cmap='Blues')
        axes[x, y].set_title(f't = {it}')

    plt.show()
