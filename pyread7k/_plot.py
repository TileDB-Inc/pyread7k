from enum import Enum
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap


def plotwireframe(amp, ax: Optional[Axes] = None, row_reduction_factor: int = 200):
    """Creates wireframe plot of the amplitude

    Args:
        amp: Input amplitude
        ax: Axis to plot on
        row_reduction_factor: Fraction to max reduce the sample space

    """
    # The block_reduce import is quite heavy and moving it into this function
    # which is less used speeds up the loading of the module
    from skimage.measure import block_reduce

    if ax is None:
        _, ax = plt.subplots()
    reduced_amp = block_reduce(amp, (row_reduction_factor, 1), np.max)

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


def plot_ping_amp_2d(
    amp: np.ndarray,
    ranges: np.ndarray,
    bearing: np.ndarray,
    ax: Optional[Axes] = None,
    cmap: Optional[Union[str, ListedColormap]] = "turbo",
    polar: Optional[bool] = False,
    interpolation: Optional[str] = "none",
    **kwargs
):  # type: ignore
    """Creates imshow plot of amplitude

    Args:
        amp: Amplitude to plot
        ranges: The sample range values
        bearing: The bearing values
        ax: Axis to plot on
        cmap: Colormap
        polar: Flag signifying whether to plot as polar or not.
               Polar is not suggested right now because matplotlib is not
               geared for it. It can produce polar plots, but it will
               generate a lot of whitespace unless you manually do some
               processing of the figure
        interpolation: The type of interpolation to use for imshow
        **kwargs: Optional arguments such as vmin, vmax, and title

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=polar)

    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)

    min_range = min(ranges)
    max_range = max(ranges)
    min_beam = min(bearing)
    max_beam = max(bearing)
    extent = [min_beam, max_beam, min_range, max_range]

    if not polar:
        ax.imshow(
            amp,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation=interpolation,
        )
    else:
        theta, rng = np.meshgrid(bearing, ranges)
        ax.pcolormesh(theta, rng, amp, cmap=cmap)
        ax.set_xlim([theta.min(), theta.max()])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    ax.set_title(kwargs.get("title", ""))
    plt.tight_layout()

    return ax
