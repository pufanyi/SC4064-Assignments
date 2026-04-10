#!/usr/bin/env python3
"""Generate stream timeline visualization showing lack of overlap."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

# Extracted from nsys trace (first 4 images, single-GPU mode)
# Times in microseconds, offset to start at 0
t0 = 497412.530  # reference start

images = [
    {  # Image 1 (stream 14)
        "H2D": (497412.530 - t0, 7.871),
        "Blur": (497694.479 - t0, 3.840),
        "Sobel": (497724.015 - t0, 2.144),
        "Histogram": (497760.751 - t0, 51.967),
        "Sync": (497812.718 - t0, 1410.837),  # host blocked
        "Equalize": (499352.322 - t0, 2.112),
        "D2H": (499360.162 - t0, 7.552),
    },
    {  # Image 2 (stream 15)
        "H2D": (499367.842 - t0, 7.936),
        "Blur": (499379.938 - t0, 3.808),
        "Sobel": (499384.898 - t0, 2.144),
        "Histogram": (499392.130 - t0, 76.063),
        "Sync": (499468.193 - t0, 19.456),
        "Equalize": (499529.345 - t0, 1.952),
        "D2H": (499534.977 - t0, 8.288),
    },
    {  # Image 3 (stream 16)
        "H2D": (499542.337 - t0, 7.360),
        "Blur": (499551.489 - t0, 3.840),
        "Sobel": (499556.353 - t0, 2.080),
        "Histogram": (499562.689 - t0, 96.319),
        "Sync": (499659.008 - t0, 17.376),
        "Equalize": (499713.983 - t0, 2.048),
        "D2H": (499718.751 - t0, 7.264),
    },
]

colors = {
    "H2D": "#3498db",
    "Blur": "#e74c3c",
    "Sobel": "#e67e22",
    "Histogram": "#2ecc71",
    "Sync": "#bdc3c7",
    "Equalize": "#9b59b6",
    "D2H": "#1abc9c",
}

fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.2))

bar_height = 0.6
y_positions = [2, 1, 0]  # Image 1 at top

for img_idx, (img, y) in enumerate(zip(images, y_positions)):
    for stage_name, (start, duration) in img.items():
        if stage_name == "Sync":
            continue
        ax.barh(
            y,
            duration,
            left=start,
            height=bar_height,
            color=colors[stage_name],
            edgecolor="black",
            linewidth=0.3,
        )

# Labels
ax.set_yticks([2, 1, 0])
ax.set_yticklabels(
    ["Image 1\n(Stream 14)", "Image 2\n(Stream 15)", "Image 3\n(Stream 16)"],
    fontsize=7.5,
)
ax.set_xlabel("Time (μs)", fontsize=9)
ax.set_title(
    "GPU Timeline — No Overlap Between Streams", fontsize=10, fontweight="bold"
)

# Legend
legend_patches = [
    mpatches.Patch(color=colors[k], label=k)
    for k in ["H2D", "Blur", "Sobel", "Histogram", "Equalize", "D2H"]
]
ax.legend(
    handles=legend_patches,
    fontsize=6.5,
    ncol=6,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
)

ax.set_xlim(-50, 2400)
ax.tick_params(labelsize=7.5)
ax.grid(True, axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig("report/stream_timeline.pdf", dpi=300, bbox_inches="tight")
plt.savefig("report/stream_timeline.png", dpi=300, bbox_inches="tight")
print("Stream timeline saved.")
