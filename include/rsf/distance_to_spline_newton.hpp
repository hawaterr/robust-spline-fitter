#pragma once

#include <vector>

#include "rsf/cardinal_spline.hpp"

namespace rsf {

// Squared distance from `query` to the closest point on one cardinal spline
// segment, found via Newton's method on d/dt ||P(t)-query||^2 = 0, t in
// [0, 1]. Falls back to comparing against the segment endpoints (t=0, t=1)
// to guard against Newton converging to a non-minimum or a degenerate
// derivative. If outT is non-null, writes the winning t.
double getClosestSquaredDistanceToSegmentUsingNewton(const Point2D& p0, const Point2D& p1, const Point2D& p2,
                                                     const Point2D& p3, const Point2D& query, double tension,
                                                     double* outT = nullptr);

// Analytic distance from `point` to the piecewise cardinal spline defined by
// sortedControlPoints (must be sorted by x, size >= 2), including its linear
// extensions. Solves getClosestSquaredDistanceToSegmentUsingNewton for every segment (cheap
// since numberOfControlPoints is small) plus a closed-form point-to-ray
// distance at both ends using the boundary segment's analytic tangent,
// matching extendSplineEndsLinearly's behavior without sampling. This is
// fitRansac's scoring function, replacing what used to be a linear scan over
// a densely-sampled curve (the previous dominant cost in fitRansac).
double getClosestDistanceToSplineUsingNewton(const Point2D& point, const std::vector<Point2D>& sortedControlPoints,
                                             double tension);

}  // namespace rsf
