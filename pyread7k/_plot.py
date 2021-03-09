from enum import Enum
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from skimage.measure import block_reduce

from ._ping_processing import Beamformed


class PingData(Enum):
    AMP = 1
    PHS = 2


def plotwireframe(amp, ax: Optional[Axes] = None, rrf: int = 200):
    if ax is None:
        _, ax = plt.subplots()
    reduced_amp = block_reduce(amp, (rrf, 1), np.max)

    for i in range(reduced_amp.shape[0])[::-1]:
        plt.fill_between(
            range(256),
            i * 8,
            np.sqrt(reduced_amp[i, :]) + i * 8,
            color="black",
            zorder=900 - i,
        )
        plt.plot(
            np.sqrt(reduced_amp[i, :]) + i * 8, color="white", lw=2.5, zorder=900 - i
        )

    plt.axis("off")
    plt.show()


def plotping2d(
    ping: Beamformed,
    ax: Optional[Axes] = None,
    cmap: Optional[Union[str, ListedColormap]] = "turbo",
    polar: Optional[bool] = False,
    interpolation: Optional[str] = "none",
    **kwargs
):  # type: ignore
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=polar)

    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)

    min_range = ping.ranges[0]
    max_range = ping.ranges[-1]
    min_beam = ping.bearings[0]
    max_beam = ping.bearings[-1]
    extent = [min_beam, max_beam, min_range, max_range]

    if not polar:
        ax.imshow(
            ping.amp,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation=interpolation,
        )
    else:
        theta, rng = np.meshgrid(ping.bearings, ping.ranges)
        ax.pcolormesh(theta, rng, ping.amp, cmap=cmap)
        ax.set_xlim([theta.min(), theta.max()])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    ax.set_title(kwargs.get("title", ""))
    plt.tight_layout()

    return ax


def saveping(filepath, pingdata, ax, cmap, **kwargs):
    ax = plotping2d(pingdata, ax, cmap, **kwargs)
    dpi = kwargs.get("dpi", 200)
    plt.savefig(filepath, dpi=dpi)
