import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional
from matplotlib.gridspec import GridSpec
from matplotlib import patches
from matplotlib.cm import Colormap
from skimage.measure import block_reduce
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from skimage import io
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from pyread7k._ping import PingDataset, PingType
import time
from ._ping import Ping
from enum import Enum

sns.set(style="white")

sns.set(style="white")

class PingData(Enum):
    AMP = 1
    PHS = 2


def plotwireframe(amp, ax: Optional[Axes] = None, rrf: int = 200):
    if ax is None:
        fig, ax = plt.subplots()
    reduced_amp = block_reduce(amp, (rrf, 1), np.max)

    for i in range(reduced_amp.shape[0])[::-1]:
        plt.fill_between(range(256), i*8, np.sqrt(reduced_amp[i, :]) + i*8, color="black", zorder=900-i)
        plt.plot(np.sqrt(reduced_amp[i, :]) + i*8, color="white", lw=2.5, zorder=900-i)

    plt.axis("off")
    sns.despine(bottom=True, left=True)
    plt.show()




def plotping2d(ping: Ping, pingdata: PingData, ax: Optional[Axes], cmap: Optional[str, Colormap] = "turbo", **kwargs):         # type: ignore
    if ax is None:
        fig, ax = plt.subplots()

    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    if pingdata == PingData.AMP:
        data = ping.amp
    elif pingdata == PingData.PHS:
        data = ping.phs
    else:
        raise ValueError("Plotting functionalities for ping data only support amplitude and phase")

    min_range = ping.range_samples[0]
    max_range = ping.range_samples[-1]
    min_beam = ping.beam_angles[0]
    max_beam = ping.beam_angles[-1]
    extent = [min_beam, max_beam, min_range, max_range]

    ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    return ax

def plotpingpolar(ping, pingdata: PingData, ax: Optional[Axes], cmap: str = plt.get_cmap("turbo"), **kwargs):
    theta, rad = np.meshgrid(ping.beam_angles, ping.range_samples)
    fig = plt.figure()
    ax = plt.add_subplot(111, polar=True)
    if pingdata == PingData.AMP:
        c = ax.pcolormesh(theta, rad, ping.amp, cmap=cmap)
    elif pingdata == PingData.PHS:
        c = ax.pcolormesh(theta, rad, ping.phs, cmap=cmap)
    else:
        raise ValueError("Plotting functionalities for ping data only support amplitude and phase")

    ax.set_xlim([theta.min(), theta.max()])
    return fig



def saveping(filepath, pingdata, ax, cmap, **kwargs):
    ax = plotping(pingdata, ax, cmap, **kwargs)
    dpi = kwargs.get("dpi", 200)
    plt.savefig(filepath, dpi=dpi)


def showpredictions(
    ping,
    boxes,
    name,
    save,
    sonar_range,
    boxprobs=None,
    highquality=False,
    mindistance=100,
):
    nx = 10
    ny = 10
    gs = GridSpec(nx, ny)
    ax_img = plt.subplot(gs[0 : nx - 1, 0 : ny - 1])
    ax_hor = plt.subplot(gs[nx - 1, 0 : ny - 1])
    ax_ver = plt.subplot(gs[0 : nx - 1, ny - 1])
    ranges, beams = ping.beamformed["amp"].shape

    ax_img.imshow(
        ping.beamformed["amp"],
        aspect="auto",
        origin="lower",
        cmap=plt.get_cmap("turbo"),
        extent=[0, beams, 0, sonar_range],
    )

    yscaler = sonar_range / ranges

    for i, box in enumerate(boxes):
        width = box.width
        height = box.height * yscaler
        x = int(box.x1 - (width // 2 + 1))
        y = int(box.y1 - (box.height // 2 + 1)) * yscaler
        if y < mindistance:
            continue
        rect = patches.Rectangle(
            (x, y), width, height, linewidth=1.1, ec=(1, 0, 1, 1), fc=(1, 0, 1, 0.2)
        )
        ax_img.add_patch(rect)
        if boxprobs is not None:
            ax_img.text(
                box.x1,
                box.y1 * yscaler,
                "Mean P: {0:4.3f}".format(boxprobs[i]["mean"]),
                color="white",
                fontsize=5,
            )
            ax_img.text(
                box.x1,
                (box.y1 - box.height // 2) * yscaler,
                "Max P: {0:4.3f}".format(boxprobs[i]["max"]),
                color="white",
                fontsize=5,
            )
            ax_img.text(
                box.x1,
                (box.y1 - box.height) * yscaler,
                "Total: {0:4.0f}".format(boxprobs[i]["count"]),
                color="white",
                fontsize=5,
            )

    ax_img.get_xaxis().set_visible(False)
    ax_ver.plot(np.mean(ping.beamformed["amp"], axis=1), range(ranges))
    ax_ver.set_ylim([0, ranges])
    ax_hor.plot(range(beams), np.mean(ping.beamformed["amp"], axis=0))
    ax_hor.set_xlim([0, beams])
    ax_ver.set_axis_off()
    ax_hor.set_axis_off()

    plt.tight_layout()

    if save:
        if highquality:
            plt.savefig(name, dpi=300)
        else:
            plt.savefig(name, dpi=100)
    else:
        plt.show()

def plot_contours(data: pd.DataFrame, means: np.ndarray, covs: np.ndarray, title: str):
    """visualize the gaussian components over the data"""
    amp = data["amp"].values.flatten()
    phs = data["phs"].values.flatten()
    plt.figure()
    plt.plot(amp, phs, 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.linspace(amp.min(), amp.max(), 1000)
    y = np.linspace(phs.min(), phs.max(), 1000)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid)

    plt.title(title)
    plt.tight_layout()
