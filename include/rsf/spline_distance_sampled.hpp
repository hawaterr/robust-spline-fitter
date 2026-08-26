#pragma once

#include <vector>

#include "rsf/cardinal_spline.hpp"

namespace rsf {

// Vertical distance from `point` to the piecewise cardinal spline: the
// absolute difference between point.y and the curve's y at point.x, found by
// linearly interpolating over `curveSamples` (must be sorted by x, e.g. from
// getCardinalSplineCurve + extendSplineEndsLinearly). Points outside the
// sampled curve's x-range are clamped to the nearest end sample. Cheaper
// than getClosestDistanceToSplineUsingNewton but only meaningful when the curve is a function of
// x.
double verticalDistanceToSpline(const Point2D& point, const std::vector<Point2D>& curveSamples);

// Brute-force nearest-sample distance from `point` to the piecewise cardinal
// spline: the minimum Euclidean distance to any point in `curveSamples`
// (e.g. from getCardinalSplineCurve + extendSplineEndsLinearly). Simplest
// and most direct metric, but only as accurate as the curve's sampling
// density, and slower than getClosestDistanceToSplineUsingNewton since it scans every sample.
double sampledDistanceToSpline(const Point2D& point, const std::vector<Point2D>& curveSamples);

}  // namespace rsf
