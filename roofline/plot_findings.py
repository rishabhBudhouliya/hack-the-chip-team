"""Render README figures from saved experiment results; requires Matplotlib and NumPy."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures"
ORIGINAL, CACHED = "#64748b", "#087f72"
INK, MUTED, GRID = "#172b4d", "#53637a", "#e2e8f0"


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": MUTED,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    trials = json.loads((ROOT / "roofline" / "agent_comparison_results.json").read_text())
    timing = json.loads((ROOT / "roofline" / "device_transfer.json").read_text())[0]
    attempts = [json.loads(line) for line in (ROOT / "roofline" / "roofline_transfer.jsonl").read_text().splitlines()]
    seed = attempts[0]["performance"]
    winner = min((row["performance"] for row in attempts if row.get("performance")),
                 key=lambda p: p["predicted_ns"])
    reference_samples = np.array(timing["kernels"]["reference"]["samples_us"])
    candidate_samples = np.array(timing["kernels"]["candidate"]["samples_us"])
    assert len(reference_samples) == len(candidate_samples) == 20

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.70, wspace=0.40)
    fig.text(0.075, 0.93, "RHS caching reduces traffic and device latency", fontsize=19, weight="bold")
    fig.text(0.075, 0.86, "M = 1024, K = 256, N = 1024  ·  float32  ·  Trainium2, LNC = 1", color=MUTED)
    panels = [
        ("HBM traffic", "MiB · simulator", seed["hbm_bytes"] / 2**20, winner["hbm_bytes"] / 2**20, 0),
        ("Predicted latency", "µs · compiler estimate", seed["predicted_ns"] / 1000, winner["predicted_ns"] / 1000, 2),
        ("Device median", "µs · one profiled capture", np.median(reference_samples), np.median(candidate_samples), 2),
    ]
    for ax, (title, unit, before, after, decimals) in zip(axes, panels):
        style(ax)
        ax.bar([0, 1], [before, after], color=[ORIGINAL, CACHED], width=0.55, zorder=3)
        ax.set_xticks([0, 1], ["Original", "RHS cache"])
        ax.set_ylim(0, before * 1.35)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", pad=34, fontsize=13)
        ax.text(0, 1.06, f"{(1-after/before)*100:.1f}% lower", transform=ax.transAxes,
                color=CACHED, fontsize=12, weight="bold")
        for index, value in enumerate([before, after]):
            ax.text(index, value + before * 0.035, f"{value:.{decimals}f}", ha="center", weight="bold")
    axes[0].axhline(6, color=INK, linestyle=(0, (3, 3)), linewidth=1)
    axes[0].text(0.5, 5.2, "Assumed minimum: 6 MiB", ha="center", fontsize=9, color=INK,
                 bbox=dict(facecolor="white", edgecolor="none", pad=2.5))
    fig.text(0.075, 0.05, "Device median: 20 timed executions per kernel; validation and 10 warmups excluded. Hardware correctness passed.",
             fontsize=10, color=MUTED)
    save(fig, "results-overview")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.23, top=0.70)
    fig.text(0.06, 0.92, "Correct improvements: worked example vs. verbal guidance", fontsize=16, weight="bold")
    fig.text(0.06, 0.84, "Qwen/Qwen3-8B  ·  4 rounds per run  ·  1 candidate per round", color=MUTED)
    for y, condition in [(1, "worked_example"), (0, "verbal_guidance_control")]:
        rows = [r for r in trials["trials"] if r["condition"] == condition]
        assert len(rows) == 3
        for x, row in enumerate(rows, start=1):
            ax.scatter(x, y, marker="o" if row["success"] else "x", s=210,
                       color=CACHED if row["success"] else ORIGINAL, linewidths=2.7, zorder=3)
        successes = sum(row["success"] for row in rows)
        ax.text(3.5, y, f"{successes}/3", va="center", fontsize=17, weight="bold")
    ax.set_yticks([1, 0], ["Worked code example", "Verbal guidance\n(control)"])
    ax.set_xticks([1, 2, 3], ["Run 1", "Run 2", "Run 3"])
    ax.set_xlim(0.6, 4)
    ax.set_ylim(-0.6, 1.6)
    ax.tick_params(length=0, colors=MUTED, pad=10)
    ax.spines[:].set_visible(False)
    fig.text(0.06, 0.07, "● Correct improvement   × No correct improvement within the budget   ·   Three runs per condition", fontsize=10, color=MUTED)
    save(fig, "agent-outcomes")

    savings = reference_samples - candidate_samples
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.24, top=0.76)
    fig.text(0.06, 0.93, "The candidate was faster in every timed pair", fontsize=17, weight="bold")
    fig.text(0.06, 0.85, "Device trace  ·  same capture and shape  ·  20 paired iterations", color=MUTED)
    style(ax)
    ax.scatter(np.arange(1, 21), savings, color=CACHED, s=38, zorder=3)
    ax.axhline(np.median(savings), color=CACHED, linestyle="--", linewidth=1.2)
    ax.text(1, 36.8, f"Median paired saving: {np.median(savings):.2f} µs", color=CACHED, weight="bold")
    ax.set_xlim(0.4, 20.6)
    ax.set_ylim(0, 40)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Timed iteration")
    ax.set_ylabel("Paired device-time saving (µs)")
    fig.text(0.06, 0.055, "Positive values favor caching. This is one hardware capture; the 20 pairs are not 20 independent captures.", fontsize=10, color=MUTED)
    save(fig, "device-latency-savings")
    print(f"Saved three figures as PNG and SVG in {OUTPUT}")


if __name__ == "__main__":
    main()
