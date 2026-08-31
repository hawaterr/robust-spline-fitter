"""Integration tests for CardinalSplineRegressor.

Each test_* function is self-contained: it names its own CSV, its own
constructor kwargs, and its own minimum-inlier bar. To add a new case,
copy one of these functions and change the three things above.

Run:
    pytest tests/                      # no plots
    pytest tests/ --plot-failures      # save a PNG for any test that fails
    pytest tests/ --plot-all           # save a PNG for every test
Plots land in tests/diagnostics/<test_name>.png
"""
from pathlib import Path

import pandas as pd

from robust_spline_fitter import CardinalSplineRegressor
from plotting import save_diagnostic_plot

DATA_DIR = Path(__file__).parent / "data"




def run_case(name, csv_path, kwargs, min_inliers, request):
    """Loads csv_path, fits the model, handles plotting, checks inlier count."""
    plot_all = request.config.getoption("--plot-all")
    plot_failures = request.config.getoption("--plot-failures")

    df = pd.read_csv(csv_path)
    x, y = df["x"].tolist(), df["y"].tolist()

    model = CardinalSplineRegressor(**kwargs)
    model.fit(x, y)

    if plot_all:
        save_diagnostic_plot(name, x, y, model)

    try:

        assert model.inlier_count_ >= min_inliers, (
            f"{name}: expected >= {min_inliers} inliers, got {model.inlier_count_}"
        )
    except AssertionError:
        if plot_failures and not plot_all:
            save_diagnostic_plot(name, x, y, model)
        raise

# tests ------------

def test_explicit(request): # request is defined by pytest, 
    run_case(
        name="test_explicit",
        csv_path=DATA_DIR / "explicit.csv",
        kwargs=dict(
            control_points=4,
            tries=1000,
            threshold=0.15,
            tension=0.2,
            samples_per_segment=200,
            min_control_point_x_gap=1.0,
            max_control_point_y_gap=2.0,
            distance_to_curve="newton",
            fit_range="data_x_range",
            curve_type="explicit",
        ),
        min_inliers=80,
        request=request,
    )

def test_vertical(request):
    run_case(
        name="test_vertical",
        csv_path=DATA_DIR / "vertical.csv",
        kwargs=dict(
            control_points=4,
            tries=1000,
            threshold=0.1,
            tension=0.4,
            samples_per_segment=200,
            min_control_point_x_gap=0,
            max_control_point_y_gap=1000.0,
            distance_to_curve="sampled",
            fit_range="inlier_x_range",
            curve_type="implicit",
        ),
        min_inliers=100,
        request=request,
    )




def test_implicit(request):
    run_case(
        name="test_implicit",
        csv_path=DATA_DIR / "implicit.csv",
        kwargs=dict(
            control_points=4,
            tries=1000,
            threshold=2,
            tension=0.4,
            samples_per_segment=200,
            min_control_point_x_gap=0,
            max_control_point_y_gap=1000,
            distance_to_curve="sampled",
            fit_range="inlier_x_range",
            curve_type="implicit"
        ),
        min_inliers=100,
        request=request,
    )
