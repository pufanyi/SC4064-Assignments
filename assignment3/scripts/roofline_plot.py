#!/usr/bin/env python3
"""Generate roofline plot for H100 SXM GPU."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# H100 SXM5 specs
peak_flops = 66.9e3  # GFLOPs/s (FP32)
peak_bw = 3350  # GB/s (HBM3)
ridge_point = peak_flops / peak_bw  # ~19.97 FLOPs/byte

# Arithmetic intensity range
ai = np.logspace(-2, 3, 500)

# Roofline
roofline = np.minimum(peak_bw * ai, peak_flops)

# Our kernels
kernels = {
    "Gaussian Blur": {"ai": 19.51, "perf": 3.50},
    "Sobel": {"ai": 1.80, "perf": 2.23},
    "Hist+Equal": {"ai": 0.45, "perf": 0.014},
}

colors = {"Gaussian Blur": "#e74c3c", "Sobel": "#3498db", "Hist+Equal": "#2ecc71"}
markers = {"Gaussian Blur": "o", "Sobel": "s", "Hist+Equal": "^"}

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))

# Roofline
ax.loglog(ai, roofline, "k-", linewidth=2, label="Roofline")

# Memory-bound and compute-bound regions
ax.axhline(y=peak_flops, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.axvline(x=ridge_point, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# Ridge point annotation
ax.annotate(
    f"Ridge point\n({ridge_point:.1f} FLOPs/B)",
    xy=(ridge_point, peak_flops),
    xytext=(ridge_point * 3, peak_flops * 0.15),
    fontsize=7.5,
    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    ha="left",
    va="top",
    color="gray",
)

# Plot kernels
for name, data in kernels.items():
    ax.plot(
        data["ai"],
        data["perf"],
        markers[name],
        color=colors[name],
        markersize=9,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=f"{name} (AI={data['ai']:.2f})",
        zorder=5,
    )

# Labels
ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=10)
ax.set_ylabel("Performance (GFLOPs/s)", fontsize=10)
ax.set_title("Roofline Model — NVIDIA H100 80GB SXM (FP32)", fontsize=10, fontweight="bold")

ax.set_xlim(0.01, 1000)
ax.set_ylim(0.001, 200000)

# Add region labels
ax.text(0.15, 300, "Memory\nBound", fontsize=8, color="#555", ha="center", style="italic")
ax.text(200, 300, "Compute\nBound", fontsize=8, color="#555", ha="center", style="italic")

# Peak annotations
ax.text(
    500,
    peak_flops * 1.15,
    f"Peak FP32: {peak_flops / 1e3:.1f} TFLOPS",
    fontsize=7.5,
    ha="right",
    va="bottom",
    color="black",
)
ax.text(
    0.012,
    peak_bw * 0.012 * 1.3,
    f"Peak BW: {peak_bw} GB/s",
    fontsize=7.5,
    ha="left",
    va="bottom",
    color="black",
    rotation=38,
)

ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax.grid(True, which="both", alpha=0.15)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("report/roofline.pdf", dpi=300, bbox_inches="tight")
plt.savefig("report/roofline.png", dpi=300, bbox_inches="tight")
print("Roofline plot saved.")
