#!/usr/bin/env python3
"""Fit a robust cardinal spline to data/data.csv using CardinalSplineRegressor."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from robust_spline_fitter import CardinalSplineRegressor

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "vertical.csv"


def main():
    df = pd.read_csv(DATA_CSV)
    x, y = df["x"].tolist(), df["y"].tolist()

    model = CardinalSplineRegressor(
        control_points=4,
        tries=5000,
        threshold=0.1,
        tension=0.4,
        samples_per_segment=200,
        min_control_point_x_gap=1.0,
        max_control_point_y_gap=2.0,
        distance_to_curve="newton",
        fit_range="inlier_x_range",
        curve_type="implicit"
    )
    model.fit(x, y)

    print(f"Loaded {len(df)} points from {DATA_CSV}")
    print(f"Best fit: {model.inlier_count_} / {len(df)} inliers using "
          f"{model.control_points} control points over {model.tries} tries")

    curve_ys = [p[1] for p in model.curve_]
    margin = 0.5 * (max(y) - min(y))
    assert all(np.isfinite([c for p in model.curve_ for c in p])), "curve has non-finite points"
    assert min(y) - margin < min(curve_ys) and max(curve_ys) < max(y) + margin, \
        "curve escapes the data's y-range"

    x_query = np.linspace(min(x), max(x), 5)
    # y_pred = model.predict(x_query)
    # print("predict() sample:")
    # for xq, yq in zip(x_query, y_pred):
    #     print(f"  x={xq:.2f} -> y={yq:.2f}")

    in_x = [xi for xi, inlier in zip(x, model.inliers_) if inlier]
    in_y = [yi for yi, inlier in zip(y, model.inliers_) if inlier]
    out_x = [xi for xi, inlier in zip(x, model.inliers_) if not inlier]
    out_y = [yi for yi, inlier in zip(y, model.inliers_) if not inlier]
    curve_x = [p[0] for p in model.curve_]
    curve_y = [p[1] for p in model.curve_]
    ctrl_x = [p[0] for p in model.control_points_]
    ctrl_y = [p[1] for p in model.control_points_]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3)

    ax.scatter(out_x, out_y, c="darkgray", s=14, alpha=0.9, label="outliers")
    ax.scatter(in_x, in_y, c="tab:blue", s=14, label="inliers")
    ax.plot(curve_x, curve_y, c="tab:red", linewidth=2.5, label="fitted spline", zorder=3)
    ax.scatter(ctrl_x, ctrl_y, c="black", marker="x", s=70, linewidth=2, label="control points", zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Robust cardinal spline fit")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
