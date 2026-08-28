#pragma once

#include <random>
#include <vector>

#include "rsf/cardinal_spline.hpp"

namespace rsf {

// How getClosestDistanceToSplineUsingNewton (and hence inlier scoring) measures a point's
// distance to the candidate curve.
enum class DistanceMetric {
    // Analytic perpendicular distance to the nearest point on the curve,
    // found via Newton's method (getClosestSquaredDistanceToSegmentUsingNewton). Correct
    // for curves that fold back in x, but costs a handful of Newton solves
    // per point.
    Perpendicular,
    // Vertical lookup: compares the point's y to the curve's y at the same
    // x, via linear interpolation over a densely sampled curve. Cheaper and
    // simpler, but only meaningful when the curve is a function of x (no
    // folding back), since it ignores any curve point at a different x.
    Vertical,
    // Brute-force nearest neighbor over a densely sampled curve: minimum
    // Euclidean distance from the point to any sample. Handles folding
    // curves like Perpendicular, but its accuracy is bounded by
    // samplesPerSegment rather than exact, and it's O(curve samples) per
    // point instead of O(control points).
    Sampled,
};

// How far the returned curve (FitResult::curve) is linearly extended at
// each end, via extendSplineEndsLinearly.
enum class FitRange {
    // Extend to cover only the winning candidate's inlier x-range. 
    Inliers,
    // Extend to cover the full input data's x-range
    Data,
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
    FitRange fitRange = FitRange::Inliers;
};

struct FitResult {
    std::vector<Point2D> controlPoints;
    std::vector<Point2D> curve;
    std::vector<bool> inlierMask;
    int inlierCount = 0;
};

// sortedControlPoints must already be sorted by x.
bool satisfiesSpacing(const std::vector<Point2D>& sortedControlPoints, double minXGap, double maxYGap);

// Robustly fits a cardinal spline to data via RANSAC: repeatedly samples
// numberOfControlPoints random data points as control points, builds the
// spline through them (extended linearly per params.fitRange), and keeps
// the candidate with the most inliers (points within `threshold` of the
// curve).
FitResult fitRansac(const std::vector<Point2D>& data, const RansacFitParams& params, std::mt19937& rng);

}  // namespace rsf
