#!/usr/bin/env python3
"""Plot the output of the fit_spline executable (run from the build dir)."""
import csv
import sys

import matplotlib.pyplot as plt


def read_csv(path):
    xs, ys, flags = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            if "inlier" in row:
                flags.append(row["inlier"] == "1")
    return xs, ys, flags


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data_points.csv"
    curve_path = sys.argv[2] if len(sys.argv) > 2 else "spline_curve.csv"
    control_path = sys.argv[3] if len(sys.argv) > 3 else "control_points.csv"

    xs, ys, inliers = read_csv(data_path)
    curve_x, curve_y, _ = read_csv(curve_path)
    ctrl_x, ctrl_y, _ = read_csv(control_path)

    in_x = [x for x, flag in zip(xs, inliers) if flag]
    in_y = [y for y, flag in zip(ys, inliers) if flag]
    out_x = [x for x, flag in zip(xs, inliers) if not flag]
    out_y = [y for y, flag in zip(ys, inliers) if not flag]

    plt.scatter(out_x, out_y, c="lightgray", s=12, label="outliers")
    plt.scatter(in_x, in_y, c="tab:blue", s=12, label="inliers")
    plt.plot(curve_x, curve_y, c="tab:red", linewidth=2, label="fitted spline")
    plt.scatter(ctrl_x, ctrl_y, c="black", marker="x", s=60, label="control points")
    plt.legend()
    plt.title("Robust cardinal spline fit")
    plt.show()


if __name__ == "__main__":
    main()
