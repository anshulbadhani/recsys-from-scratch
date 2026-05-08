"""
recsys_figures.py
=================
Generates three publication-ready figures from the RecSys report:
  1. Inference latency breakdown (doughnut)
  2. Amdahl's Law – observed vs theoretical speedup
  3. Recall@10 vs Catalog Coverage scatter

Run:  python recsys_figures.py
Output: figures/fig1_latency_pie.pdf  (+ .png)
        figures/fig2_amdahl.pdf       (+ .png)
        figures/fig3_recall_coverage.pdf (+ .png)

Swap the DATA sections below with real measurements if available.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "legend.handlelength":2.0,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

# Palette (from report colour system, accessible)
C_BLUE   = "#185FA5"
C_TEAL   = "#0F6E56"
C_CORAL  = "#993C1D"
C_GRAY   = "#888780"
C_AMBER  = "#BA7517"
C_PURPLE = "#534AB7"
C_PINK   = "#993556"
C_GREEN  = "#3B6D11"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Latency Pie (doughnut)
# ─────────────────────────────────────────────────────────────────────────────

# DATA – from Table 10 of the report
STAGES = [
    "KD-Tree query",
    "Bloom filter",
    "Ranking (MMR)",
    "User embedding",
    "History lookup",
    "ASIN conversion",
]
CYCLES   = [8_000_000, 580_000, 580_000, 15_000, 5_000, 10_000]  # avg cycles
COLORS_P = [C_BLUE, C_TEAL, C_CORAL, C_GRAY, C_AMBER, C_PURPLE]

total   = sum(CYCLES)
pcts    = [c / total * 100 for c in CYCLES]
latency_us = [c / 3.6e9 * 1e6 for c in CYCLES]   # µs at 3.6 GHz

fig, ax = plt.subplots(figsize=(6.0, 4.6))
fig.subplots_adjust(left=0.02, right=0.58, top=0.88, bottom=0.06)

wedges, _ = ax.pie(
    pcts,
    colors=COLORS_P,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.2),
    radius=1.0,
)

# Centre annotation
ax.text(0, 0.08, "0.33 ms",  ha="center", va="center",
        fontsize=12, fontweight="bold", color="#1a1916")
ax.text(0, -0.22, "per user", ha="center", va="center",
        fontsize=8,  color="#555")

# Legend table on the right
legend_items = []
for stage, pct, lat, col in zip(STAGES, pcts, latency_us, COLORS_P):
    patch = mpatches.Patch(color=col, label=f"{stage}  {pct:.1f}%  ({lat:.0f} µs)")
    legend_items.append(patch)

leg = ax.legend(
    handles=legend_items,
    loc="center left",
    bbox_to_anchor=(1.08, 0.5),
    title="Stage  |  Share  |  Latency",
    title_fontsize=8.5,
    fontsize=8.5,
    frameon=True,
    handlelength=1.2,
    handleheight=1.0,
    borderpad=0.7,
    labelspacing=0.55,
)
leg._legend_box.align = "left"

ax.set_title(
    "Per-stage inference latency breakdown\n"
    r"$D{=}64$, bipolar weights, OpenMP $\times$14 cores",
    fontsize=10, pad=10, loc="left"
)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig1_latency_pie.{ext}")
print("Saved fig1_latency_pie")
plt.show()
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Amdahl's Law
# ─────────────────────────────────────────────────────────────────────────────

# DATA – serial fraction derived from the report (Section 5.5)
F_SERIAL   = 0.23          # serial fraction
N_CORES    = 14            # physical cores
OBS_CORES  = 14
OBS_SPEED  = 3.55          # observed speedup

# Optimisation milestones (Table 8)
MILESTONES = [
    (1,  1.00,  "Baseline\n(serial)"),
    (1,  3.34,  "Compiler\n-O3"),
    (1,  3.68,  "Candidate\nreduction"),
    (1,  7.56,  "AVX2\nSIMD"),
    (14, 26.8,  "OpenMP\n×14"),
    (14, 354,   "PCA\n384D→64D"),
]

cores_range = np.linspace(1, 16, 400)
speedup_amdahl = 1.0 / (F_SERIAL + (1 - F_SERIAL) / cores_range)
speedup_ideal  = cores_range

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14,
                    wspace=0.32)

# ── Left panel: Amdahl curve ──────────────────────────────────────────────
ax = axes[0]
ax.plot(cores_range, speedup_ideal,   "--", color="#cccccc",  lw=1.2,
        label="Linear ideal")
ax.plot(cores_range, speedup_amdahl,  "-",  color=C_BLUE,     lw=2.0,
        label=rf"Amdahl ($f={F_SERIAL}$)")
ax.scatter([OBS_CORES], [OBS_SPEED],  color=C_CORAL, s=60, zorder=5,
           label=f"Observed ({OBS_SPEED}×)")

# Annotate saturation ceiling
ceiling = 1 / F_SERIAL
ax.axhline(ceiling, color=C_BLUE, lw=0.8, ls=":", alpha=0.6)
ax.text(2.2, ceiling + 0.08, rf"$1/f \approx {ceiling:.2f}$×",
        fontsize=8, color=C_BLUE, va="bottom")

# Annotate observed point
ax.annotate(f"3.55× @ 14 cores",
            xy=(OBS_CORES, OBS_SPEED),
            xytext=(9.5, 2.3),
            fontsize=8, color=C_CORAL,
            arrowprops=dict(arrowstyle="-|>", color=C_CORAL, lw=0.9),
            ha="center")

ax.set_xlabel("Number of CPU cores")
ax.set_ylabel("Speedup (×)")
ax.set_xlim(1, 16)
ax.set_ylim(0, 6.5)
ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f×"))
ax.legend(loc="upper left", frameon=True)
ax.set_title("Amdahl's Law: OpenMP parallelism", fontsize=10, pad=6, loc="left")

# ── Right panel: optimisation waterfall ──────────────────────────────────
ax2 = axes[1]
labels_wf  = ["Baseline", "+Compiler\n(-O3)", "+Candidate\nreduction",
               "+AVX2\nSIMD", "+OpenMP\n(×14)", "+PCA\n(384→64)"]
speedups   = [1.0, 3.34, 3.68, 7.56, 26.8, 354.0]
bar_colors = [C_GRAY, C_TEAL, C_TEAL, C_BLUE, C_GREEN, C_CORAL]

x = np.arange(len(labels_wf))
bars = ax2.bar(x, speedups, color=bar_colors, width=0.6, edgecolor="white",
               linewidth=0.7)

for bar, sp in zip(bars, speedups):
    label = f"{sp:.0f}×" if sp >= 10 else f"{sp:.2f}×"
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             label, ha="center", va="bottom", fontsize=7.5, color="#333")

ax2.set_xticks(x)
ax2.set_xticklabels(labels_wf, fontsize=7.8)
ax2.set_ylabel("Cumulative speedup (×)")
ax2.set_ylim(0, 420)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f×"))
ax2.set_title("Optimisation pipeline: cumulative speedup", fontsize=10, pad=6, loc="left")

fig.suptitle("Parallel scaling and optimisation progression",
             fontsize=11, y=0.97, x=0.02, ha="left")

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig2_amdahl.{ext}")
print("Saved fig2_amdahl")
plt.show()
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Recall@10 vs Catalog Coverage
# ─────────────────────────────────────────────────────────────────────────────

# DATA – from Table 7 / Table 9
CONFIGS = [
    # (label,                    recall, coverage, color,    marker, msize, group)
    ("Popularity baseline",       6.50,   0.01,  C_GRAY,   "^",  90,  "baseline"),
    ("Cosine Sort 384D",          1.68,  19.46,  C_BLUE,   "o",  70,  "384D"),
    ("Adaptive MMR 384D",         1.18,  37.28,  C_BLUE,   "s",  60,  "384D"),
    ("Cosine Sort 64D",           1.46,  17.16,  C_TEAL,   "o",  70,  "64D-norm"),
    ("Adaptive MMR 64D",          0.81,  26.37,  C_TEAL,   "s",  60,  "64D-norm"),
    ("Cosine Sort 64D signed",    1.24,  70.42,  C_CORAL,  "o",  70,  "64D-sign"),
    ("Adaptive MMR 64D signed",   0.67,  81.18,  C_CORAL,  "s",  60,  "64D-sign"),
]

fig, ax = plt.subplots(figsize=(7.8, 5.2))
fig.subplots_adjust(left=0.10, right=0.72, top=0.88, bottom=0.12)

# Draw a faint Pareto frontier line connecting non-dominated points
# (manually identified: higher coverage always trades recall)
pareto = sorted(
    [(c[2], c[1]) for c in CONFIGS],   # (coverage, recall)
    key=lambda p: p[0]
)
px, py = zip(*pareto)
ax.plot(px, py, "--", color="#cccccc", lw=1.0, zorder=1, label="_nolegend_")

for label, recall, cov, color, marker, msize, group in CONFIGS:
    ax.scatter(cov, recall, color=color, marker=marker, s=msize,
               edgecolors="white", linewidths=0.8, zorder=4)
    # Offset annotations to avoid overlap
    offsets = {
        "Popularity baseline":     ( 1.2, 0.18),
        "Cosine Sort 384D":        ( 1.0, 0.12),
        "Adaptive MMR 384D":       ( 1.0,-0.22),
        "Cosine Sort 64D":         (-1.0,-0.24),
        "Adaptive MMR 64D":        ( 1.0, 0.12),
        "Cosine Sort 64D signed":  (-1.0,-0.24),
        "Adaptive MMR 64D signed": ( 1.0, 0.12),
    }
    dx, dy = offsets.get(label, (1.0, 0.12))
    ax.annotate(
        label,
        xy=(cov, recall),
        xytext=(cov + dx * 3.5, recall + dy),
        fontsize=7.5,
        color="#444",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        va="center",
    )

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_GRAY,
           markersize=8, label="Popularity baseline"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BLUE,
           markersize=7, label="Cosine Sort"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BLUE,
           markersize=7, label="Adaptive MMR"),
    mpatches.Patch(color="white", label=""),  # spacer
    mpatches.Patch(facecolor=C_BLUE,  label=r"384D, normalised $w$"),
    mpatches.Patch(facecolor=C_TEAL,  label=r"64D PCA, normalised $w$"),
    mpatches.Patch(facecolor=C_CORAL, label=r"64D PCA, signed $w \in [-1,1]$"),
]
ax.legend(handles=legend_elements, loc="upper left",
          bbox_to_anchor=(1.03, 1.0),
          title="Configuration",
          title_fontsize=8.5,
          fontsize=8.5, frameon=True,
          borderpad=0.8, labelspacing=0.5)

ax.set_xlabel(r"Catalog coverage @ $K=10$ (%)")
ax.set_ylabel(r"Recall @ $K=10$ (%)")
ax.set_xlim(-3, 90)
ax.set_ylim(-0.3, 8.0)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])

# Quadrant shading (low coverage / high recall vs ideal zone)
ax.axvspan(-3,  20, alpha=0.04, color=C_CORAL, zorder=0)
ax.axvspan(60,  90, alpha=0.04, color=C_TEAL,  zorder=0)
ax.text(3,  7.3, "Filter\nbubble\nzone",  fontsize=7.5, color=C_CORAL, alpha=0.7, ha="left")
ax.text(63, 7.3, "High\ncoverage\nzone", fontsize=7.5, color=C_TEAL,  alpha=0.7, ha="left")

ax.set_title(
    "Recall@10 vs Catalog Coverage across system configurations\n"
    r"Amazon Software dataset, $N=146{,}980$ users, $|I|=89{,}247$ items, $K=10$",
    fontsize=10, pad=8, loc="left"
)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig3_recall_coverage.{ext}")
print("Saved fig3_recall_coverage")
plt.show()
plt.close()

print("\nAll figures saved to ./figures/")