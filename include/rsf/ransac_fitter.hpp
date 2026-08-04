#pragma once

#include <random>
#include <vector>

#include "rsf/cardinal_spline.hpp"

namespace rsf {

// How distanceToSpline (and hence inlier scoring) measures a point's
// distance to the candidate curve.
enum class DistanceMetric {
    // Analytic perpendicular distance to the nearest point on the curve,
    // found via Newton's method (closestDistanceSquaredOnSegment). Correct
    // for curves that fold back in x, but costs a handful of Newton solves
    // per point.
    Perpendicular,
    // Vertical lookup: compares the point's y to the curve's y at the same
    // x, via linear interpolation over a densely sampled curve. Cheaper and
    // simpler, but only meaningful when the curve is a function of x (no
    // folding back), since it ignores any curve point at a different x.
    Vertical,
};

struct RansacFitParams {
    int numberOfControlPoints = 4;
    int tries = 500;
    double tension = 0.5;
    double threshold = 0.2;
    int samplesPerSegment = 200;
    double minControlPointXGap = 1.0;
    double maxControlPointYGap = 2.0;
    DistanceMetric distanceMetric = DistanceMetric::Perpendicular;
};

struct FitResult {
    std::vector<Point2D> controlPoints;
    std::vector<Point2D> curve;
    std::vector<bool> inlierMask;
    int inlierCount = 0;
};

// Analytic distance from `point` to the piecewise cardinal spline defined by
// sortedControlPoints (must be sorted by x, size >= 2), including its linear
// extensions. Solves closestDistanceSquaredOnSegment for every segment (cheap
// since numberOfControlPoints is small) plus a closed-form point-to-ray
// distance at both ends using the boundary segment's analytic tangent,
// matching extendSplineEndsLinearly's behavior without sampling. This is
// fitRansac's scoring function, replacing what used to be a linear scan over
// a densely-sampled curve (the previous dominant cost in fitRansac).
double distanceToSpline(const Point2D& point, const std::vector<Point2D>& sortedControlPoints, double tension);

// Vertical distance from `point` to the piecewise cardinal spline: the
// absolute difference between point.y and the curve's y at point.x, found by
// linearly interpolating over `curveSamples` (must be sorted by x, e.g. from
// getCardinalSplineCurve + extendSplineEndsLinearly). Points outside the
// sampled curve's x-range are clamped to the nearest end sample. Cheaper
// than distanceToSpline but only meaningful when the curve is a function of
// x.
double verticalDistanceToSpline(const Point2D& point, const std::vector<Point2D>& curveSamples);

// sortedControlPoints must already be sorted by x.
bool satisfiesSpacing(const std::vector<Point2D>& sortedControlPoints, double minXGap, double maxYGap);

// Robustly fits a cardinal spline to data via RANSAC: repeatedly samples
// numberOfControlPoints random data points as control points, builds the
// spline through them (extended linearly to cover data's own x-range), and
// keeps the candidate with the most inliers (points within `threshold` of
// the curve).
FitResult fitRansac(const std::vector<Point2D>& data, const RansacFitParams& params, std::mt19937& rng);

}  // namespace rsf
