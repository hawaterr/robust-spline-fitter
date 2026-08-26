"""Python-friendly, sklearn-style wrapper around the compiled rsf extension."""
from typing import List, Sequence, Tuple

import numpy as np

try:
    import rsf
except ImportError as e:
    raise ImportError(
        "The compiled 'rsf' extension was not found. If you're working from "
        "a source checkout (not `pip install`-ed), build it first: "
        "cmake -S . -B build && cmake --build build."
    ) from e


class CardinalSplineRegressor:
    """Robust cardinal-spline regressor, fit via RANSAC.

    Fits a smooth cardinal spline to noisy 2D (x, y) data. Candidate
    splines are built from randomly sampled control points and scored by
    how many data points fall within `threshold` of the curve; the
    best-scoring candidate is kept. The search itself runs in compiled C++
    (see rsf.fit_ransac), so it stays fast even with many `tries`.

    Parameters
    ----------
    control_points : int, default=4
        Number of control points sampled per candidate spline. More control
        points can follow sharper curves but need more data to constrain
        them well.
    tension : float, default=0.5
        Cardinal spline tension in [0, 1]. 0 gives a Catmull-Rom spline
        (loose, rounded corners); 1 flattens the tangents (straighter
        segments between control points).
    tries : int, default=50000
        Number of random control-point samples to try. Higher is more
        likely to find the best fit, at a roughly linear cost in runtime.
    threshold : float, default=0.2
        Max distance from the curve for a data point to count as an
        inlier, in the same units as x/y.
    samples_per_segment : int, default=200
        How finely each spline segment is sampled internally. Higher is
        more accurate (better distance-to-curve estimates) but slower.
    min_control_point_x_gap : float, default=1.0
        Reject any candidate where two adjacent (x-sorted) control points
        are within this x distance of each other. Avoids numerically
        unstable, near-degenerate spline segments.
    max_control_point_y_gap : float, default=2.0
        Reject any candidate where two adjacent (x-sorted) control points
        differ in y by more than this. Avoids wild swings between control
        points that happen to land close together in x.
    distance_metric : {"perpendicular", "vertical", "sampled"}, default="perpendicular"
        Inlier calculation method: how a data point's distance to a
        candidate curve is measured for inlier scoring. "perpendicular" is
        the analytic nearest-point distance (via Newton's method) and
        handles curves that fold back in x. "vertical" is a cheaper lookup
        that compares the point's y to the curve's y at the same x (linear
        interpolation over the sampled curve); only meaningful when the
        curve is a function of x. "sampled" is a brute-force nearest
        neighbor search over the densely sampled curve; handles folding
        curves like "perpendicular" but its accuracy is bounded by
        samples_per_segment and it's slower.
    seed : int, default=42
        Seed for the random control-point sampling, for reproducible fits.

    Attributes
    ----------
    Set by `fit`, following the sklearn convention of a trailing
    underscore for attributes populated during fitting (as opposed to a
    leading underscore, which conventionally marks a "private" attribute
    not meant to be read by callers at all):

    control_points_ : list of (float, float)
        The winning candidate's control points.
    curve_ : list of (float, float)
        Sampled points along the fitted spline, x-ordered, including the
        linear extension out to the data's own x-range.
    inliers_ : list of bool
        Per-input-point inlier mask against the winning curve, aligned
        with the `x`/`y` passed to `fit`.
    inlier_count_ : int
        Number of True values in `inliers_`.

    Examples
    --------
    >>> model = CardinalSplineRegressor(control_points=4, tries=50000)
    >>> model.fit(x, y)
    >>> y_pred = model.predict(x_query)
    >>> model.inliers_
    """

    def __init__(
        self,
        control_points: int = 4,
        tension: float = 0.5,
        tries: int = 5000,
        threshold: float = 0.2,
        samples_per_segment: int = 200,
        min_control_point_x_gap: float = 1.0,
        max_control_point_y_gap: float = 2.0,
        distance_metric: str = "perpendicular",
        seed: int = 42,
    ):
        self.control_points = control_points
        self.tension = tension
        self.tries = tries
        self.threshold = threshold
        self.samples_per_segment = samples_per_segment
        self.min_control_point_x_gap = min_control_point_x_gap
        self.max_control_point_y_gap = max_control_point_y_gap
        self.distance_metric = distance_metric
        self.seed = seed

        self.control_points_: List[Tuple[float, float]] = []
        self.curve_: List[Tuple[float, float]] = []
        self.inliers_: List[bool] = []
        self.inlier_count_: int = 0

    def fit(self, x: Sequence[float], y: Sequence[float]) -> "CardinalSplineRegressor":
        """Fit the spline to (x, y) data via RANSAC.

        Parameters
        ----------
        x, y : sequence of float
            The data points to fit, same length.

        Returns
        -------
        self, so calls can be chained, e.g. `model = CardinalSplineRegressor().fit(x, y)`.
        """
        metric_by_name = {
            "perpendicular": rsf.DistanceMetric.Perpendicular,
            "vertical": rsf.DistanceMetric.Vertical,
            "sampled": rsf.DistanceMetric.Sampled,
        }
        if self.distance_metric not in metric_by_name:
            raise ValueError(
                f"distance_metric must be one of {sorted(metric_by_name)}, got {self.distance_metric!r}"
            )

        params = rsf.RansacFitParams()
        params.numberOfControlPoints = self.control_points
        params.tries = self.tries
        params.threshold = self.threshold
        params.tension = self.tension
        params.samplesPerSegment = self.samples_per_segment
        params.minControlPointXGap = self.min_control_point_x_gap
        params.maxControlPointYGap = self.max_control_point_y_gap
        params.distanceMetric = metric_by_name[self.distance_metric]

        result = rsf.fit_ransac(list(x), list(y), params, seed=self.seed)

        self.control_points_ = result.control_points
        self.curve_ = result.curve
        self.inliers_ = result.inlier_mask
        self.inlier_count_ = result.inlier_count
        return self

    def predict(self, x_query: Sequence[float]) -> np.ndarray:
        """Interpolate the fitted spline at the given x values.

        Must be called after `fit`.

        Parameters
        ----------
        x_query : sequence of float
            x values to evaluate the fitted spline at.

        Returns
        -------
        numpy.ndarray of the interpolated y values, same length as `x_query`.
        """
        if not self.curve_:
            raise RuntimeError("CardinalSplineRegressor.predict() called before fit()")
        curve_x, curve_y = zip(*self.curve_)
        return np.interp(x_query, curve_x, curve_y)
