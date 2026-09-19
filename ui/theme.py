"""
One design system for the live dashboard and every report figure (Session I).

The values are the data-viz reference palette, used unchanged, so every rule below is
a checked one rather than a taste:

  categorical   a FIXED order of eight hues, one step per mode. Validated with the
                skill's validate_palette.js on the first four slots (the most any
                figure here uses): worst adjacent colour-vision-deficiency Delta E
                9.1 light / 8.4 dark (target >= 8), normal-vision 22.9 / 19.8
                (floor 15). In LIGHT mode slots 3 and 4 (aqua, yellow) sit below
                3:1 on the surface, so a figure that uses them labels them directly.
  sequential    one hue, blue, light to dark: magnitude. On the dark surface the
                ramp runs the other way so small values recede into the surface.
  status        good / warning / serious / critical, reserved for STATE and never
                used for a series; always drawn with an icon and a label, so a
                state is never carried by colour alone.
  ink           text wears ink tokens (primary, secondary, muted), never a series
                colour; gridlines and axes recede.

Light is for reports and PDFs, dark for the live dashboard.
"""

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "success": "#006300",
         "neutral": "#f0efec"}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "success": "#0ca30c",
        "neutral": "#383835"}

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SERIES_NAMES = ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")

# the single-hue sequential ramp, steps 100 .. 700
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
ICON = {"good": "✓", "warning": "⚠", "serious": "▲", "critical": "✕"}

FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
ICON_FONT = ["Segoe UI Symbol", "DejaVu Sans"]      # the UI sans has no check or cross glyph


def tokens(mode="light"):
    if mode not in ("light", "dark"):
        raise ValueError("mode must be 'light' or 'dark'")
    return LIGHT if mode == "light" else DARK


def series(i, mode="light"):
    """Categorical slot i (0-based) in its fixed order. There is no ninth colour: fold or facet."""
    pal = SERIES[mode]
    if not 0 <= i < len(pal):
        raise IndexError(f"categorical slot {i} does not exist; fold the rest into 'Other' or facet")
    return pal[i]


def sequential_cmap(mode="light", name="blue"):
    """Magnitude: one hue. Low values sit nearest the surface in either mode."""
    steps = BLUE_RAMP if mode == "light" else BLUE_RAMP[::-1]
    return LinearSegmentedColormap.from_list(f"{name}_{mode}", steps)


def apply(mode="light"):
    """Set matplotlib's defaults to the system's tokens for `mode`; returns the tokens."""
    t = tokens(mode)
    mpl.rcParams.update({
        "figure.facecolor": t["page"], "savefig.facecolor": t["page"], "axes.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8, "axes.labelcolor": t["ink2"],
        "axes.titlecolor": t["ink"], "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 8, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=SERIES[mode]),
        "xtick.color": t["axis"], "ytick.color": t["axis"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 5, "patch.linewidth": 0.0,
        "font.family": "sans-serif", "font.sans-serif": FONT + list(mpl.rcParams["font.sans-serif"]),
        "font.size": 9, "text.color": t["ink"],
        "legend.fontsize": 8, "legend.frameon": False, "legend.labelcolor": t["ink2"],
        "figure.titlesize": 12, "figure.titleweight": "semibold",
    })
    return t


def status_line(fig, items, x=0.01, y=0.012, mode="light", fontsize=9, sep="     "):
    """
    A row of states at figure coordinates (x, y): each item is (state, label) with state
    in STATUS; the icon wears the status colour and the label the primary ink. Returns
    the artist so a live figure can remove and redraw it.
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    t = tokens(mode)
    boxes = []
    for i, (state, label) in enumerate(items):
        if i:
            boxes.append(TextArea(sep, textprops={"fontsize": fontsize}))
        if state is not None:
            boxes.append(TextArea(ICON[state] + " ", textprops={"color": STATUS[state], "fontsize": fontsize + 1,
                                                                "fontfamily": ICON_FONT}))
        boxes.append(TextArea(label, textprops={"color": t["ink"] if state else t["ink2"], "fontsize": fontsize}))
    box = AnchoredOffsetbox(loc="lower left", child=HPacker(children=boxes, align="baseline", pad=0, sep=0),
                            pad=0.0, frameon=False, bbox_to_anchor=(x, y), bbox_transform=fig.transFigure,
                            borderpad=0.0)
    fig.add_artist(box)
    return box
