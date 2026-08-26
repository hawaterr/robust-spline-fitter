#!/usr/bin/env python3
"""Interactively click out (x, y) points and save them to data/<name>.csv.

Usage:
    python3 scripts/generate_data.py
    python3 scripts/generate_data.py --xlim -5 5 --ylim -2 2

Left-click to add a point, right-click to undo the last one. Close the plot
window when done; you'll be prompted for a filename to save under data/.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlim", type=float, nargs=2, default=(-10.0, 10.0), metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", type=float, nargs=2, default=(-10.0, 10.0), metavar=("YMIN", "YMAX"))
    args = parser.parse_args()

    points = []

    fig, ax = plt.subplots()
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_title(
        f"x in {args.xlim}, y in {args.ylim} | "
        "Left-click: add point | Right-click: undo | Close window when done"
    )
    scatter = ax.scatter([], [], c="tab:blue", s=20)

    def redraw():
        if points:
            xs, ys = zip(*points)
        else:
            xs, ys = [], []
        scatter.set_offsets(list(zip(xs, ys)))
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # left click: add
            points.append((event.xdata, event.ydata))
        elif event.button == 3 and points:  # right click: undo
            points.pop()
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if not points:
        print("No points were added, nothing to save.")
        return

    name = input(f"Save {len(points)} points as data/<name>.csv - name: ").strip()
    if not name:
        print("No name given, discarding points.")
        return
    if not name.endswith(".csv"):
        name += ".csv"

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / name

    with open(out_path, "w") as f:
        f.write("x,y\n")
        for x, y in points:
            f.write(f"{x},{y}\n")

    print(f"Saved {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()

# example command: python3 scripts/generate_data.py --xlim -5 5 --ylim -2 2