#!/usr/bin/env python3
"""Generate heatmap and surface-plot visualizations from wave2d snapshots."""

import os
import struct

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DX = DY = 0.01
DT = 0.005
OUT_DIR = "out"


def read_field(filename: str):
    """Read a binary snapshot: [int32 Nx][int32 Ny][Nx*Ny float64]."""
    with open(filename, "rb") as f:
        Nx = struct.unpack("i", f.read(4))[0]
        Ny = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, dtype=np.float64, count=Nx * Ny)
    return data.reshape(Ny, Nx), Nx, Ny


def plot_heatmap(data, Nx, Ny, title, filename):
    x = np.linspace(0, (Nx - 1) * DX, Nx)
    y = np.linspace(0, (Ny - 1) * DY, Ny)
    vmax = max(abs(data.min()), abs(data.max()), 1e-12)
    plt.figure(figsize=(5.5, 4.5))
    plt.pcolormesh(x, y, data, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="u(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_surface(data, Nx, Ny, title, filename):
    x = np.linspace(0, (Nx - 1) * DX, Nx)
    y = np.linspace(0, (Ny - 1) * DY, Ny)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, data, cmap="RdBu_r", linewidth=0, antialiased=True, rstride=2, cstride=2
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    steps = [0, 50, 100, 150, 200]
    for step in steps:
        binfile = os.path.join(OUT_DIR, f"field_step{step}.bin")
        if not os.path.exists(binfile):
            print(f"  {binfile} not found, skipping")
            continue
        data, Nx, Ny = read_field(binfile)
        t = step * DT
        label = f"t = {t:.3f} s"
        plot_heatmap(
            data,
            Nx,
            Ny,
            f"Wave field — {label}",
            os.path.join(OUT_DIR, f"heatmap_step{step}.pdf"),
        )
        plot_surface(
            data,
            Nx,
            Ny,
            f"Wave field — {label}",
            os.path.join(OUT_DIR, f"surface_step{step}.pdf"),
        )
        print(f"  Plotted step {step} ({label})")

    # Combined heatmap panel (for the report)
    fig, axes = plt.subplots(
        1, 5, figsize=(20, 3.6), gridspec_kw={"wspace": 0.05, "right": 0.92}
    )
    for ax, step in zip(axes, steps):
        binfile = os.path.join(OUT_DIR, f"field_step{step}.bin")
        if not os.path.exists(binfile):
            continue
        data, Nx, Ny = read_field(binfile)
        x = np.linspace(0, (Nx - 1) * DX, Nx)
        y = np.linspace(0, (Ny - 1) * DY, Ny)
        vmax = 1.0
        im = ax.pcolormesh(
            x, y, data, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax
        )
        ax.set_title(f"t = {step * DT:.2f} s")
        ax.set_xlabel("x")
        if step == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
        ax.set_aspect("equal")
    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="u(x, y)")
    fig.savefig(os.path.join(OUT_DIR, "heatmap_panel.pdf"), dpi=200)
    plt.close(fig)
    print("  Saved combined heatmap panel")


if __name__ == "__main__":
    main()
