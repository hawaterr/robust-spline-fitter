#pragma once

#include <random>
#include <vector>

#include "rsf/cardinal_spline.hpp"

namespace rsf {

struct RansacFitParams {
    int numberOfControlPoints = 4;
    int tries = 500;
    double tension = 0.5;
    double threshold = 0.2;
    int samplesPerSegment = 200;
    double minControlPointXGap = 1.0;
    double maxControlPointYGap = 2.0;
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

// sortedControlPoints must already be sorted by x.
bool satisfiesSpacing(const std::vector<Point2D>& sortedControlPoints, double minXGap, double maxYGap);

// Robustly fits a cardinal spline to data via RANSAC: repeatedly samples
// numberOfControlPoints random data points as control points, builds the
// spline through them (extended linearly to cover data's own x-range), and
// keeps the candidate with the most inliers (points within `threshold` of
// the curve).
FitResult fitRansac(const std::vector<Point2D>& data, const RansacFitParams& params, std::mt19937& rng);

}  // namespace rsf
