# robust-spline-fitter

A Python library, implemented in C++ for speed, that robustly fits a cardinal spline to noisy 2D `(x, y)` data using RANSAC. 2D only for now — see [ROADMAP.md](ROADMAP.md) for what's planned.

## Install

From source:

```
pip install .
```

This builds the C++ extension via CMake and installs the `robust_spline_fitter` package. You'll need a C++17 compiler and CMake 3.16+ available.

## Quick start

```python
from robust_spline_fitter import CardinalSplineRegressor

model = CardinalSplineRegressor(control_points=4, tries=50000)
model.fit(x, y)  # x, y: sequences of floats

y_pred = model.predict(x_query)
print(model.inlier_count_, "/", len(x), "inliers")
```

See [examples/example.py](examples/example.py) for a full runnable example, including plotting the fit.

## How it works

Candidate splines are built from randomly sampled control points (RANSAC), and each candidate is scored by how many of the input points fall within `threshold` distance of the curve. The best-scoring candidate is kept. Because the search runs in compiled C++, it stays fast even with a large number of `tries`.

## Key parameters

- `tension` — cardinal spline tension in `[0, 1]`. `0` gives a Catmull-Rom spline (loose, rounded corners); `1` flattens the tangents.
- `threshold` — max distance from the curve for a point to count as an inlier.
- `distance_metric` — `"perpendicular"` (default, analytic nearest-point distance via Newton's method, handles curves that fold back in x), `"vertical"` (cheaper y-at-same-x lookup, only valid when the curve is a function of x), or `"sampled"` (brute-force nearest-sample search).

Full parameter docs are in the `CardinalSplineRegressor` docstring.

## License

Non-commercial use only (personal, educational, research) — no AI/ML training use, attribution required, no warranty. See [LICENSE.txt](LICENSE.txt) for full terms. For commercial licensing, contact ali.hawater@gmail.com.
