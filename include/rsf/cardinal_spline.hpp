#pragma once

#include <vector>

#include "rsf/point.hpp"

namespace rsf {

// Evaluates one segment of a cardinal spline between p1 and p2, using the
// neighboring control points p0 and p3 to derive tangents. t in [0, 1].
// tension in [0, 1]: 0 gives a Catmull-Rom spline, 1 flattens tangents to zero.
Point2D evalCardinalSegment(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3, double t,
                            double tension);

// First and second derivatives of evalCardinalSegment with respect to t.
// Used by getClosestSquaredDistanceToSegmentUsingNewton for Newton's method.
Point2D evalCardinalSegmentDerivative(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3,
                                      double t, double tension);
Point2D evalCardinalSegmentSecondDerivative(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3,
                                            double t, double tension);

// Samples a full cardinal spline through an ordered list of control points.
// Phantom points are synthesized at both ends so the curve passes through
// every control point, including the first and last.
// Requires at least 2 control points.
std::vector<Point2D> getCardinalSplineCurve(const std::vector<Point2D>& controlPoints, double tension,
                                            int samplesPerSegment);

// Extends a sampled curve at both ends with straight lines, so that it
// covers [xMin, xMax]. Each extension continues the slope between the
// curve's two outermost points at that end. No-op at an end if the curve
// already reaches past xMin/xMax there. Mirrors regression.py's
// CardinalSplineRegressor._extend_splines_with_linear_assumption.
std::vector<Point2D> extendSplineEndsLinearly(const std::vector<Point2D>& curve, double xMin, double xMax,
                                              int samplesPerExtension = 200);

}  // namespace rsf
