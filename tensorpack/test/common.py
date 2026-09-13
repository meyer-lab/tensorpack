"""
This file contains functions that are used in multiple figures.
"""

import xml.etree.ElementTree as ET
from string import ascii_lowercase

import drawsvg as draw
import matplotlib
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35


def getSetup(figsize, gridd, multz=None, empts=None):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_lowercase[ii],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )


def _read_svg(path):
    """Parse an SVG file, returning its root attributes and inner markup."""
    root = ET.parse(path).getroot()
    inner = "".join(ET.tostring(child, encoding="unicode") for child in root)
    return root.attrib, inner


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1):
    """Add cartoon to a figure file."""

    # Overlay Figure cartoons
    fig_attrib, fig_inner = _read_svg(figFile)
    _, cartoon_inner = _read_svg(cartoonFile)

    d = draw.Drawing(
        fig_attrib.get("width"),
        fig_attrib.get("height"),
        viewBox=fig_attrib.get("viewBox"),
    )
    d.append(draw.Raw(fig_inner))

    cartoon = draw.Group(
        transform=f"translate({x}, {y}) scale({scalee * scale_x} {scalee * scale_y})"
    )
    cartoon.append(draw.Raw(cartoon_inner))
    d.append(cartoon)

    d.save_svg(figFile)
