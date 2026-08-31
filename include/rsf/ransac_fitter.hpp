#pragma once

#include <limits>
#include <random>
#include <vector>

#include "rsf/cardinal_spline.hpp"

namespace rsf {

// How getClosestDistanceToSplineUsingNewton (and hence inlier scoring) measures a point's
// distance to the candidate curve.
enum class DistanceMetric {
    // Analytic distance to the nearest point on the curve, found via Newton's
    // method (getClosestSquaredDistanceToSegmentUsingNewton). Correct for
    // curves that fold back in x, but costs a handful of Newton solves per
    // point.
    Newton,
    // Vertical lookup: compares the point's y to the curve's y at the same
    // x, via linear interpolation over a densely sampled curve. Cheaper and
    // simpler, but only meaningful when the curve is a function of x (no
    // folding back), since it ignores any curve point at a different x.
    Vertical,
    // Brute-force nearest neighbor over a densely sampled curve: minimum
    // Euclidean distance from the point to any sample. Handles folding
    // curves like Newton, but its accuracy is bounded by
    // samplesPerSegment rather than exact, and it's O(curve samples) per
    // point instead of O(control points).
    Sampled,
};

// What shape the data is. This decides how the randomly sampled control points
// are ordered along the curve before the spline is built through them.
enum class CurveType {
    // y is a function of x: a single y at each x. Control points are sorted by
    // x, which is cheap, but a candidate that folds back in x can never be
    // represented.
    Explicit,
    // One x may have several y, so the curve can fold back on itself and run
    // near-vertically. Control points are ordered so the total path through
    // them is shortest (an open TSP, via orderPointsByProximity), at the cost
    // of an exact solve per try. Pair with DistanceMetric::Newton or
    // ::Sampled, the only orientation-free metrics.
    Implicit,
};

// How far the returned curve (FitResult::curve) is linearly extended at
// each end, via extendSplineEndsLinearly.
enum class FitRange {
    // Extend to cover only the winning candidate's inlier x-range.
    InlierXRange,
    // Extend to cover the full input data's x-range
    DataXRange,
};

struct RansacFitParams {
    int numberOfControlPoints = 4;
    int tries = 500;
    double tension = 0.5;
    double threshold = 0.2;
    int samplesPerSegment = 200;
    double minControlPointXGap = std::numeric_limits<double>::quiet_NaN();
    double maxControlPointYGap = std::numeric_limits<double>::quiet_NaN();
    double minControlPointDGap = std::numeric_limits<double>::quiet_NaN();
    DistanceMetric distanceMetric = DistanceMetric::Newton;
    FitRange fitRange = FitRange::InlierXRange;
    CurveType curveType = CurveType::Explicit;
};

struct FitResult {
    std::vector<Point2D> controlPoints;
    std::vector<Point2D> curve;
    std::vector<bool> inlierMask;
    int inlierCount = 0;
};

// controlPoints must already be in curve order (whichever curve type produced
// them). Any constraint passed as NaN is skipped, so a candidate with no
// constraints set is always accepted.
//
// minXGap is a *signed* x gap and maxYGap a y cap, both of which assume the
// control points are ordered along x - true only under Explicit. minDGap is a
// straight-line distance, which needs no ordering, so it is the one constraint
// that also works under Implicit. Passing the x/y pair under Implicit is a
// caller error the Python wrapper rejects before it gets here.
bool satisfiesSpacing(const std::vector<Point2D>& controlPoints, double minXGap, double maxYGap, double minDGap,
                      CurveType curveType);

// Robustly fits a cardinal spline to data via RANSAC: repeatedly samples
// numberOfControlPoints random data points as control points, builds the
// spline through them (extended linearly per params.fitRange), and keeps
// the candidate with the most inliers (points within `threshold` of the
// curve).
FitResult fitRansac(const std::vector<Point2D>& data, const RansacFitParams& params, std::mt19937& rng);

}  // namespace rsf
