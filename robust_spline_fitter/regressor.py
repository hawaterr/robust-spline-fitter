"""Python-friendly, sklearn-style wrapper around the compiled rsf extension."""
from typing import List, Sequence, Tuple

import numpy as np

try:
    from . import rsf
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
    ordering : {"xsorted", "distance"}, default="xsorted"
        How the sampled control points are ordered along the curve.
        "xsorted" sorts them by x, which is cheap but assumes the data is a
        function of x (single y per x), so a curve that folds back in x can
        never be represented. "distance" instead orders them so the total
        path through them is shortest (an open TSP, solved exactly), which
        makes folding and near-vertical curves representable. Pair
        "distance" with distance_metric="perpendicular", the only
        orientation-free metric, and note that `predict` is unavailable on a
        curve that folds back - use `predict_at_u` instead.

        Expect "distance" to be slower, typically by more than the ordering
        itself costs. The exact solve is small at the default 4 control
        points (~0.4us per try, ~20ms over tries=50000) though it grows
        sharply with control points (~46us each at 8, ~284us at 10). The
        larger effect is that "distance" applies min_control_point_x_gap as
        a straight-line gap and skips the max_control_point_y_gap cap, so
        many more candidates survive spacing and go on to be scored - which
        is where the time actually goes. Raising min_control_point_x_gap
        brings it back down.
    fit_range : {"inliers", "data"}, default="inliers"
        How far the fitted curve (`curve_`) is linearly extended at each
        end. "inliers" extends only to cover the winning candidate's own
        inlier x-range, so outliers far away in x can't drag the curve's
        ends out. "data" extends to cover the full input x-range regardless
        of which points ended up as inliers.
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
        fit_range: str = "inliers",
        ordering: str = "xsorted",
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
        self.fit_range = fit_range
        self.ordering = ordering
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

        range_by_name = {
            "inliers": rsf.FitRange.Inliers,
            "data": rsf.FitRange.Data,
        }
        if self.fit_range not in range_by_name:
            raise ValueError(f"fit_range must be one of {sorted(range_by_name)}, got {self.fit_range!r}")

        ordering_by_name = {
            "xsorted": rsf.ControlPointOrdering.XSorted,
            "distance": rsf.ControlPointOrdering.Distance,
        }
        if self.ordering not in ordering_by_name:
            raise ValueError(f"ordering must be one of {sorted(ordering_by_name)}, got {self.ordering!r}")

        params = rsf.RansacFitParams()
        params.numberOfControlPoints = self.control_points
        params.tries = self.tries
        params.threshold = self.threshold
        params.tension = self.tension
        params.samplesPerSegment = self.samples_per_segment
        params.minControlPointXGap = self.min_control_point_x_gap
        params.maxControlPointYGap = self.max_control_point_y_gap
        params.distanceMetric = metric_by_name[self.distance_metric]
        params.fitRange = range_by_name[self.fit_range]
        params.ordering = ordering_by_name[self.ordering]

        result = rsf.fit_ransac(list(x), list(y), params, seed=self.seed)

        self.control_points_ = result.control_points
        self.curve_ = result.curve
        self.inliers_ = result.inlier_mask
        self.inlier_count_ = result.inlier_count
        return self

    def predict(self, x_query: Sequence[float]) -> np.ndarray:
        """Interpolate the fitted spline at the given x values.

        Must be called after `fit`. Only valid when the fitted curve is a
        function of x; a curve that folds back has several y values at one
        x, so use `predict_at_u` for those instead.

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
        # np.interp needs increasing x and silently returns nonsense without
        # it, so refuse rather than hand back a wrong number.
        if any(b < a for a, b in zip(curve_x, curve_x[1:])):
            raise RuntimeError(
                "the fitted curve is not a function of x (it folds back), so "
                "predict() has no single y per x. Use predict_at_u() instead."
            )
        return np.interp(x_query, curve_x, curve_y)

    def predict_at_u(self, u_query: Sequence[float]) -> np.ndarray:
        """Evaluate the fitted spline at the given curve parameters.

        Must be called after `fit`. Unlike `predict`, this stays well-defined
        for curves that fold back in x, so it is the way to evaluate a fit
        made with ordering="distance".

        `u` runs over [0, n-1] for n control points: the integer part picks
        the segment and the fractional part is that segment's parameter, so
        u=1.5 is halfway along segment 1 and integer u lands exactly on a
        control point. Note the spacing is in the spline's own parameter,
        not arc length, so equal steps in u are not equally spaced in
        distance. u is clamped to [0, n-1].

        This evaluates the spline between the control points only; it does
        not include the linear end extensions carried by `curve_`.

        Parameters
        ----------
        u_query : sequence of float
            Curve parameters to evaluate at.

        Returns
        -------
        numpy.ndarray of shape (len(u_query), 2), the (x, y) curve points.
        """
        if not self.control_points_:
            raise RuntimeError("CardinalSplineRegressor.predict_at_u() called before fit()")
        points = rsf.eval_spline_at_u(self.control_points_, list(u_query), self.tension)
        return np.array(points, dtype=float)
