"""Shared helper for saving diagnostic plots of a spline fit."""
import matplotlib
matplotlib.use("Agg")  # headless: never opens a window, works in CI
import matplotlib.pyplot as plt
from pathlib import Path

DIAG_DIR = Path(__file__).parent / "diagnostics"
DIAG_DIR.mkdir(exist_ok=True)


def save_diagnostic_plot(name, x, y, model):

    inliers = model.inliers_
    in_x = [xi for xi, inlier in zip(x, inliers) if inlier]
    in_y = [yi for yi, inlier in zip(y, inliers) if inlier]
    out_x = [xi for xi, inlier in zip(x, inliers) if not inlier]
    out_y = [yi for yi, inlier in zip(y, inliers) if not inlier]
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
    ax.scatter(ctrl_x, ctrl_y, c="black", marker="x", s=70, linewidth=2,
               label="control points", zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(name)
    fig.tight_layout()

    out_path = DIAG_DIR / f"{name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
