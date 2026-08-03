#!/usr/bin/env python3
"""Fit a robust cardinal spline to data/data.csv using CardinalSplineRegressor."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spline_regressor import CardinalSplineRegressor


def main():
    df = pd.read_csv("data/data.csv")
    x, y = df["x"].tolist(), df["y"].tolist()

    model = CardinalSplineRegressor(
        control_points=4,
        tries=50000,
        threshold=0.2,
        tension=0.5,
        samples_per_segment=200,
        min_control_point_x_gap=1.0,
        max_control_point_y_gap=2.0,
    )
    model.fit(x, y)

    print(f"Loaded {len(df)} points from data/data.csv")
    print(f"Best fit: {model.inlier_count_} / {len(df)} inliers using "
          f"{model.control_points} control points over {model.tries} tries")

    x_query = np.linspace(min(x), max(x), 5)
    y_pred = model.predict(x_query)
    print("predict() sample:")
    for xq, yq in zip(x_query, y_pred):
        print(f"  x={xq:.2f} -> y={yq:.2f}")

    in_x = [xi for xi, inlier in zip(x, model.inliers_) if inlier]
    in_y = [yi for yi, inlier in zip(y, model.inliers_) if inlier]
    out_x = [xi for xi, inlier in zip(x, model.inliers_) if not inlier]
    out_y = [yi for yi, inlier in zip(y, model.inliers_) if not inlier]
    curve_x = [p[0] for p in model.curve_]
    curve_y = [p[1] for p in model.curve_]
    ctrl_x = [p[0] for p in model.control_points_]
    ctrl_y = [p[1] for p in model.control_points_]

    plt.scatter(out_x, out_y, c="lightgray", s=12, label="outliers")
    plt.scatter(in_x, in_y, c="tab:blue", s=12, label="inliers")
    plt.plot(curve_x, curve_y, c="tab:red", linewidth=2, label="fitted spline")
    plt.scatter(ctrl_x, ctrl_y, c="black", marker="x", s=60, label="control points")
    plt.legend()
    plt.title("Robust cardinal spline fit")
    plt.show()


if __name__ == "__main__":
    main()
