#pragma once

#include <vector>

#include "rsf/point.hpp"

namespace rsf {

// Returns controlPoints[i], synthesizing a phantom point by linear reflection
// when i falls before the first or after the last control point, so boundary
// segments still have a well-defined p0/p3.
Point2D selectControlPoint(const std::vector<Point2D>& controlPoints, int i);

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

// Evaluates the spline at parameter u in [0, n-1] for n control points: the
// integer part picks the segment, the fractional part is that segment's
// Hermite parameter t. So u = 3.5 is halfway (in t, not in distance) along
// segment 3, and integer u lands exactly on a control point.
//
// Unlike a lookup by x this stays well-defined when the curve folds back in x.
// u is clamped to [0, n-1]. Requires at least 2 control points.
Point2D evalCardinalSplineAtU(const std::vector<Point2D>& controlPoints, double u, double tension);

// Extends a sampled curve at both ends with straight lines, so that it
// covers [xMin, xMax]. Each extension continues the slope between the
// curve's two outermost points at that end. No-op at an end if the curve
// already reaches past xMin/xMax there. Mirrors regression.py's
// CardinalSplineRegressor._extend_splines_with_linear_assumption.
std::vector<Point2D> extendSplineEndsLinearly(const std::vector<Point2D>& curve, double xMin, double xMax,
                                              int samplesPerExtension = 200);

// Extension for curves that may fold back in x, where "the end of the curve"
// and "the end nearest xMin" are not the same thing. Each end continues along
// the spline's analytic tangent there, in the direction the curve is actually
// pointing as it leaves - the same outward rays
// getClosestDistanceToSplineUsingNewton scores against, so the drawn curve and
// the scored curve agree.
//
// Each extension stops at the furthest `inliers` point projected onto that
// end's own outward direction, so it ends at the last inlier it actually runs
// past. An x-range bound cannot express this on a folding curve: min/max x is
// one box shared by both ends, and the x-extreme inlier may belong to the
// opposite end of the curve.
//
// Kept separate from extendSplineEndsLinearly because that one steps along x
// from each end outward, which silently reverses an extension whose tangent
// points away from the bound it was aimed at. Under CurveType::Explicit
// control points are sorted by x, so the tangents can only point outward and
// the two agree; only folding curves need this.
std::vector<Point2D> extendSplineEndsAlongTangent(const std::vector<Point2D>& curve,
                                                  const std::vector<Point2D>& controlPoints,
                                                  const std::vector<Point2D>& inliers, double tension,
                                                  int samplesPerExtension = 200);

}  // namespace rsf
