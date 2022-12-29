_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0
_PAD_INCHES = 0.015

palette = [
    "#377eb8",
    "#ff7f00",
    "#984ea3",
    "#4daf4a",
    "#e41a1c",
    "brown",
    "green",
    "red",
]
MARKERS = [
    "^-",
    "1-",
    "*-",
    "s-",
    "+-",
    "o-",
    ">-",
    "d-",
    "2-",
    "3-",
    "4-",
    "8-",
    "<-",
]


base_fontstyle = {
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.titlesize": 8,
}
base_tex = {
    "text.usetex": False,
}
base_fontfamily = {
    "text.usetex": False,
    "font.sans-serif": ["Arial"],
    "font.family": "sans-serif",
}
base_other = {
    "lines.markersize": 4,
    "figure.dpi": 150,
}
base_layout = {
    "figure.constrained_layout.use": True,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": _PAD_INCHES,
}
base_style = {
    **base_fontfamily,
    **base_fontstyle,
    **base_tex,
    **base_other,
    **base_layout,
}


def figsize(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=1,
    height_to_width_ratio=_GOLDEN_RATIO,
):
    """Neurips figure size defaults."""
    base_width_in = 5.5
    width_in = base_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    figsize = width_in, height_in
    return figsize


LINEWIDTH = 1
