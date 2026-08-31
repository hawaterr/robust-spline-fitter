"""Python-friendly, sklearn-style wrapper around the compiled rsf extension."""
from typing import List, Optional, Sequence, Tuple

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
    min_control_point_x_gap : float, optional
        Reject any candidate where two adjacent (x-sorted) control points
        are within this x distance of each other. Avoids numerically
        unstable, near-degenerate spline segments. Only valid under
        curve_type="explicit", since it is a signed gap that assumes the
        control points run left to right.
    max_control_point_y_gap : float, optional
        Reject any candidate where two adjacent (x-sorted) control points
        differ in y by more than this. Avoids wild swings between control
        points that happen to land close together in x. Only valid under
        curve_type="explicit", where it would otherwise forbid exactly the
        near-vertical segments an implicit curve exists to allow.
    min_control_point_d_gap : float, optional
        Reject any candidate where two adjacent control points are within
        this straight-line distance of each other. Carries no direction, so
        unlike the two above it stays meaningful on a curve that folds back
        or runs vertically - it is the only spacing constraint accepted
        under curve_type="implicit".

        The three constraints default to None, meaning unset: a fit with
        none of them set applies no spacing filter at all. Because a
        straight-line gap and an x/y gap are alternative ways to space
        control points, min_control_point_d_gap cannot be combined with
        either of the other two; `fit` raises ValueError if it is.
    distance_to_curve : {"newton", "vertical", "sampled"}, default="newton"
        Inlier calculation method: how a data point's distance to a
        candidate curve is measured for inlier scoring. "newton" is the
        analytic nearest-point distance (solved with Newton's method) and
        handles curves that fold back in x. "vertical" is a cheaper lookup
        that compares the point's y to the curve's y at the same x (linear
        interpolation over the sampled curve); only meaningful when the
        curve is a function of x. "sampled" is a brute-force nearest
        neighbor search over the densely sampled curve; handles folding
        curves like "newton" but its accuracy is bounded by
        samples_per_segment and it's slower.
    curve_type : {"explicit", "implicit"}, default="explicit"
        The shape of the data. "explicit" means y is a function of x - a
        single y at each x. "implicit" allows several y at one x, so the
        curve can fold back on itself and run near-vertically.

        This decides how the sampled control points are ordered before the
        spline is built through them: "explicit" sorts them by x, which is
        cheap but makes a folding curve unrepresentable; "implicit" orders
        them so the total path through them is shortest (an open TSP,
        solved exactly).

        Because it sets which curves can be fitted, the other options have
        to be compatible with it. "implicit" rejects the ones that assume a
        single y per x: distance_to_curve="vertical" (undefined when one x
        has several y, so use "newton" or "sampled") and
        fit_range="data_x_range" (an implicit curve isn't laid out along x).
        It also rejects min_control_point_x_gap and max_control_point_y_gap,
        which assume x-ordered control points; space them with
        min_control_point_d_gap instead. Note that `predict` is unavailable
        on a curve that folds back - use `predict_at_u` instead.

        Expect "implicit" to be slower, typically by more than the ordering
        itself costs. The exact solve is small at the default 4 control
        points (~0.4us per try, ~20ms over tries=50000) though it grows
        sharply with control points (~46us each at 8, ~284us at 10). The
        larger effect is that the spacing changes above let many more
        candidates survive and go on to be scored - which is where the time
        actually goes. Raising min_control_point_d_gap brings it back down.
    fit_range : {"inlier_x_range", "data_x_range"}, default="inlier_x_range"
        How far the fitted curve (`curve_`) is linearly extended at each
        end. "inlier_x_range" extends only to cover the winning candidate's
        own inlier x-range, so outliers far away in x can't drag the
        curve's ends out. "data_x_range" extends to cover the full input
        x-range regardless of which points ended up as inliers. Extension
        runs along x, which an implicit curve isn't laid out along, so only
        the default is accepted under curve_type="implicit".
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
        linear extension out to the data's own x-range. Under
        curve_type="implicit" the points follow the curve rather than x.
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
        min_control_point_x_gap: Optional[float] = None,
        max_control_point_y_gap: Optional[float] = None,
        min_control_point_d_gap: Optional[float] = None,
        distance_to_curve: str = "newton",
        fit_range: str = "inlier_x_range",
        curve_type: str = "explicit",
        seed: int = 42,
    ):
        self.control_points = control_points
        self.tension = tension
        self.tries = tries
        self.threshold = threshold
        self.samples_per_segment = samples_per_segment
        self.min_control_point_x_gap = min_control_point_x_gap
        self.max_control_point_y_gap = max_control_point_y_gap
        self.min_control_point_d_gap = min_control_point_d_gap
        self.distance_to_curve = distance_to_curve
        self.fit_range = fit_range
        self.curve_type = curve_type
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
            "newton": rsf.DistanceMetric.Newton,
            "vertical": rsf.DistanceMetric.Vertical,
            "sampled": rsf.DistanceMetric.Sampled,
        }
        if self.distance_to_curve not in metric_by_name:
            raise ValueError(
                f"distance_to_curve must be one of {sorted(metric_by_name)}, got {self.distance_to_curve!r}"
            )

        range_by_name = {
            "inlier_x_range": rsf.FitRange.InlierXRange,
            "data_x_range": rsf.FitRange.DataXRange,
        }
        if self.fit_range not in range_by_name:
            raise ValueError(f"fit_range must be one of {sorted(range_by_name)}, got {self.fit_range!r}")

        curve_type_by_name = {
            "explicit": rsf.CurveType.Explicit,
            "implicit": rsf.CurveType.Implicit,
        }
        if self.curve_type not in curve_type_by_name:
            raise ValueError(f"curve_type must be one of {sorted(curve_type_by_name)}, got {self.curve_type!r}")

        # Spacing constraints. The straight-line gap and the x/y pair are two
        # ways of saying "keep control points apart", so allowing both at once
        # would leave it ambiguous which one a rejected candidate failed;
        # require the caller to pick one.
        xy_given = [
            name
            for name, value in (
                ("min_control_point_x_gap", self.min_control_point_x_gap),
                ("max_control_point_y_gap", self.max_control_point_y_gap),
            )
            if value is not None
        ]
        if self.min_control_point_d_gap is not None and xy_given:
            raise ValueError(
                "min_control_point_d_gap cannot be combined with "
                f"{' or '.join(xy_given)}: a straight-line gap and an x/y gap are "
                "alternative ways to space control points, so set one or the other."
            )

        if self.curve_type == "implicit":
            # An implicit curve can have several y at one x, so options that
            # assume a single y per x are not just suboptimal here, they are
            # wrong - refuse rather than fit something plausible-looking off
            # a meaningless setting.
            if xy_given:
                raise ValueError(
                    f"{' and '.join(xy_given)} "
                    f"{'assume' if len(xy_given) > 1 else 'assumes'} control points ordered "
                    'along x, which is not the case under curve_type="implicit" (the curve may '
                    "fold back or run vertically). Use min_control_point_d_gap, a straight-line "
                    "gap, instead."
                )
            if self.distance_to_curve == "vertical":
                raise ValueError(
                    'distance_to_curve="vertical" measures y at a given x, which is undefined '
                    'when one x can have several y. Use distance_to_curve="newton" (or "sampled") '
                    'with curve_type="implicit".'
                )
            if self.fit_range != "inlier_x_range":
                raise ValueError(
                    f'fit_range={self.fit_range!r} has no meaning with curve_type="implicit": an '
                    "implicit curve is not laid out along x, so extending it to cover the whole "
                    'data x-range is not well defined. Use fit_range="inlier_x_range" (the default).'
                )

        params = rsf.RansacFitParams()
        params.numberOfControlPoints = self.control_points
        params.tries = self.tries
        params.threshold = self.threshold
        params.tension = self.tension
        params.samplesPerSegment = self.samples_per_segment
        # NaN is how the C++ side spells "constraint not set": every comparison
        # against it is false, so it never rejects a candidate.
        unset = float("nan")
        params.minControlPointXGap = unset if self.min_control_point_x_gap is None else self.min_control_point_x_gap
        params.maxControlPointYGap = unset if self.max_control_point_y_gap is None else self.max_control_point_y_gap
        params.minControlPointDGap = unset if self.min_control_point_d_gap is None else self.min_control_point_d_gap
        params.distanceMetric = metric_by_name[self.distance_to_curve]
        params.fitRange = range_by_name[self.fit_range]
        params.curveType = curve_type_by_name[self.curve_type]

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
        made with curve_type="implicit".

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
