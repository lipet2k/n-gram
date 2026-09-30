"""Figures: search cost against cache memory, and steps by query length. PNG only."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

INK = "#1f1f1f"
INK_2 = "#5f5e5a"
GRID = "#e6e5e1"
NONE = "#8a8984"
BLUE, GREEN, YELLOW, PINK = "#3d94dc", "#4fae3c", "#d4a900", "#dd6fc3"
FILL = {BLUE: "#a8d4f5", GREEN: "#b4e6a4", YELLOW: "#f8e58e", PINK: "#f6bde5"}
FIXED = [BLUE, GREEN, YELLOW]
GREEDY = PINK
DEPTH_NAMES = {1: "unigram", 2: "bigram", 3: "trigram"}


def depth_name(depth: int) -> str:
    return DEPTH_NAMES.get(depth, f"{depth}-gram")


def format_bytes(value: float, _position=None) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000:
            return f"{value:.3g} {unit}"
        value /= 1000
    return f"{value:.3g} TB"


def greedy_label(floor: int) -> str:
    return (
        "greedy from the root"
        if floor == 0
        else f"all {depth_name(floor)}s, then greedy"
    )


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK_2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "text.color": INK,
            "savefig.dpi": 220,
        }
    )


def mean_steps(entry: dict, n: str) -> float:
    return entry["steps"][n]["mean"]


def log_ticks(low: float, high: float, mantissas: tuple[int, ...]) -> list[float]:
    """The values m * 10^e inside [low, high] for each mantissa m."""
    decades = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
    ticks = [m * 10.0**e for e in decades for m in mantissas]
    return [t for t in ticks if low <= t <= high]


def draw_pareto(r: dict, stem: Path) -> None:
    """Mean steps against cache memory on log axes.

    The greedy sweep from the lowest floor is the frontier line, other floors are hollow
    markers on it, and each fixed cache is tied to the frontier at its own memory. The
    right axis relabels the same scale as the speedup over no cache.
    """
    n = str(r["latency"]["query_length"])
    baseline = mean_steps(r["none"], n)
    fixed_bytes = {f["bytes"] for f in r["fixed"]}
    curves = sorted(r["greedy"], key=lambda c: c["floor_depth"])
    frontier = sorted(curves[0]["points"], key=lambda p: p["bytes"])
    at_budget = {p["budget"]: mean_steps(p, n) for p in frontier}
    points = [p for c in curves for p in c["points"]] + r["fixed"]
    xs = [p["bytes"] for p in points]
    ys = [mean_steps(p, n) for p in points if mean_steps(p, n) > 0]
    low, high = min(ys) / 1.3, baseline * 1.9

    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(xs) / 1.5, max(xs) * 2.3)
    ax.set_ylim(low, high)

    ax.axhline(
        baseline, color=NONE, linestyle=(0, (4, 3)), linewidth=1.1, label="no cache"
    )
    ax.annotate(
        "no cache",
        (ax.get_xlim()[0], baseline),
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
    )
    for i, fixed in enumerate(r["fixed"]):
        x, y = fixed["bytes"], mean_steps(fixed, n)
        color = FIXED[i % len(FIXED)]
        ax.plot(
            [x],
            [y],
            linestyle="none",
            marker="D",
            markersize=6.5,
            markerfacecolor=FILL[color],
            markeredgecolor=color,
            markeredgewidth=1.3,
            zorder=5,
            label=fixed["name"],
        )
        ratio = y / at_budget[x] if x in at_budget else 1.0
        tie = None
        if ratio >= 2:
            tie = f"{ratio:.1f}× fewer steps\nat the same memory"
        elif ratio >= 1.05:
            tie = f"{ratio:.1f}×"
        if tie:
            ax.plot([x, x], [y, at_budget[x]], color=INK_2, linewidth=0.9, zorder=2)
            ax.annotate(
                tie,
                (x, math.sqrt(y * at_budget[x])),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.2,
                color=INK_2,
                linespacing=1.15,
            )
        left = tie is not None and "\n" in tie
        ax.annotate(
            fixed["name"],
            (x, y),
            xytext=(-7 if left else 7, 4),
            textcoords="offset points",
            ha="right" if left else "left",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.plot(
        [p["bytes"] for p in frontier],
        [mean_steps(p, n) for p in frontier],
        color=GREEDY,
        linewidth=1.9,
        label=greedy_label(curves[0]["floor_depth"]),
        zorder=3,
    )
    swept = [p for p in frontier if p["budget"] not in fixed_bytes]
    ax.plot(
        [p["bytes"] for p in swept],
        [mean_steps(p, n) for p in swept],
        linestyle="none",
        marker="o",
        markersize=4.5,
        color=GREEDY,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )
    for curve in curves[1:]:
        swept = [p for p in curve["points"] if p["budget"] not in fixed_bytes]
        ax.plot(
            [p["bytes"] for p in swept],
            [mean_steps(p, n) for p in swept],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=GREEDY,
            markeredgewidth=1.4,
            label=greedy_label(curve["floor_depth"]),
            zorder=5,
        )
    ax.axvline(
        r["index_bytes"], color=GRID, linewidth=1.0, linestyle=(0, (1, 2)), zorder=1
    )
    ax.annotate(
        f"index on disk: {format_bytes(r['index_bytes'])}",
        (r["index_bytes"], high / 1.08),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        color=INK_2,
    )

    ax.xaxis.set_major_formatter(FuncFormatter(format_bytes))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(log_ticks(low, high, (1, 3))))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("cache memory")
    ax.set_ylabel(f"mean search steps per {n}-token query")

    def speedup(steps):
        return baseline / np.maximum(np.asarray(steps, dtype=float), 1e-9)

    right = ax.secondary_yaxis("right", functions=(speedup, speedup))
    right.yaxis.set_major_locator(
        FixedLocator(log_ticks(max(baseline / high, 1), baseline / low, (1, 2, 5)))
    )
    right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    right.yaxis.set_minor_locator(NullLocator())
    right.spines["right"].set_visible(False)
    right.tick_params(axis="y", length=0)

    ax.set_title("Search cost versus memory", pad=16)
    ax.annotate(
        f"{r['corpus']}-token corpus, one step is two random reads",
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
    )
    ax.legend(loc="lower left")
    save(fig, stem)


def draw_by_length(r: dict, stem: Path) -> None:
    """Mean steps against query length for no cache, each fixed cache, and greedy at the largest fixed size."""
    lengths = r["query_lengths"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    ax.plot(
        lengths,
        [r["none"]["steps"][str(n)]["mean"] for n in lengths],
        color=NONE,
        linestyle=(0, (4, 3)),
        label="no cache",
    )
    for i, fixed in enumerate(r["fixed"]):
        ys = [fixed["steps"][str(n)]["mean"] for n in lengths]
        ax.plot(
            lengths,
            ys,
            color=FIXED[i % len(FIXED)],
            marker="o",
            markeredgecolor="white",
            label=f"{fixed['name']} ({format_bytes(fixed['bytes'])})",
        )
    largest = max(r["fixed"], key=lambda f: f["bytes"])
    for i, curve in enumerate(r["greedy"]):
        match = [p for p in curve["points"] if p["budget"] == largest["bytes"]]
        if match:
            ys = [match[0]["steps"][str(n)]["mean"] for n in lengths]
            label = f"{greedy_label(curve['floor_depth'])} ({format_bytes(match[0]['bytes'])})"
            ax.plot(
                lengths,
                ys,
                color=GREEDY,
                marker="o",
                markerfacecolor=GREEDY if i == 0 else "white",
                markeredgecolor="white" if i == 0 else GREEDY,
                label=label,
            )
    ax.set_xticks(lengths)
    ax.set_xlabel("query length in tokens")
    ax.set_ylabel("average search steps")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_title("Steps by query length")
    ax.legend(loc="upper right")
    save(fig, stem)


def save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def draw_all(r: dict, results_dir: Path) -> None:
    style()
    draw_pareto(r, results_dir / "pareto")
    draw_by_length(r, results_dir / "steps_by_length")
