#include "rsf/ransac_fitter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "rsf/distance_to_spline_newton.hpp"
#include "rsf/distance_to_spline_other.hpp"
#include "rsf/point.hpp"

namespace rsf {

namespace {

// Draws numberOfControlPoints distinct indices from `indices` (via partial
// Fisher-Yates on a copy) and returns the corresponding data points sorted by
// x, ready to serve as spline control points.
std::vector<Point2D> sampleControlPoints(const std::vector<Point2D>& data, std::vector<int> indices,
                                         int numberOfControlPoints, std::mt19937& rng) {
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(numberOfControlPoints);
    // Order control points along the curve by x. This assumes the curve is a
    // function of x (single y per x), which holds for this synthetic test data
    // but won't hold in general (e.g. closed loops or non-monotonic curves).
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return data[a].x < data[b].x; });

    std::vector<Point2D> controlPoints;
    controlPoints.reserve(numberOfControlPoints);
    for (int idx : indices) controlPoints.push_back(data[idx]);
    return controlPoints;
}

// Builds the densely-sampled, linearly-extended curve needed by the
// Vertical/Sampled metrics. Perpendicular scores directly off control points
// and never calls this, so it's skipped there to avoid the sampling cost.
std::vector<Point2D> buildCandidateCurve(const std::vector<Point2D>& controlPoints, const RansacFitParams& params,
                                         double dataXMin, double dataXMax) {
    if (params.distanceMetric != DistanceMetric::Vertical && params.distanceMetric != DistanceMetric::Sampled) {
        return {}; // no need to build the whole curve if we use the newton method
    }
    std::vector<Point2D> curve = getCardinalSplineCurve(controlPoints, params.tension, params.samplesPerSegment);
    return extendSplineEndsLinearly(curve, dataXMin, dataXMax);
}

// Scores one candidate against all of `data` using the configured distance
// metric, returning a per-point inlier mask plus the total inlier count.
std::pair<std::vector<bool>, int> scoreCandidate(const std::vector<Point2D>& data,
                                                 const std::vector<Point2D>& controlPoints,
                                                 const std::vector<Point2D>& candidateCurve,
                                                 const RansacFitParams& params) {
    std::vector<bool> inlierMask(data.size());
    int inlierCount = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        double dist;
        switch (params.distanceMetric) {
            case DistanceMetric::Vertical:
                dist = verticalDistanceToSpline(data[i], candidateCurve);
                break;
            case DistanceMetric::Sampled:
                dist = sampledDistanceToSpline(data[i], candidateCurve);
                break;
            default:
                dist = getClosestDistanceToSplineUsingNewton(data[i], controlPoints, params.tension);
                break;
        }
        const bool isInlier = dist < params.threshold;
        inlierMask[i] = isInlier;
        if (isInlier) ++inlierCount;
    }
    return {std::move(inlierMask), inlierCount};
}

// Picks the x-range the final curve is extended to, per params.fitRange:
// either the winning candidate's own inlier x-range, or the full input
// data's x-range.
std::pair<double, double> getDataRange(const std::vector<Point2D>& data, const FitResult& best,
                                         const RansacFitParams& params, double dataXMin, double dataXMax) {
    if (params.fitRange != FitRange::Inliers) {
        return {dataXMin, dataXMax};
    }
    std::vector<Point2D> inliers;
    inliers.reserve(best.inlierCount);
    for (size_t i = 0; i < data.size(); ++i) {
        if (best.inlierMask[i]) inliers.push_back(data[i]);
    }
    return dataXRange(inliers);
}

}  // namespace

bool satisfiesSpacing(const std::vector<Point2D>& sortedControlPoints, double minXGap, double maxYGap) {
    for (size_t i = 0; i + 1 < sortedControlPoints.size(); ++i) {
        const double xGap = sortedControlPoints[i + 1].x - sortedControlPoints[i].x;
        const double yGap = std::abs(sortedControlPoints[i + 1].y - sortedControlPoints[i].y);
        if (xGap <= minXGap || yGap >= maxYGap) {
            return false;
        }
    }
    return true;
}

FitResult fitRansac(const std::vector<Point2D>& data, const RansacFitParams& params, std::mt19937& rng) {
    const auto t0 = std::chrono::steady_clock::now();
    FitResult best;

    const auto [dataXMin, dataXMax] = dataXRange(data);

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);  // Cpp: fills indices with [0,1,2,...], .begin() returns an iterator

    for (int attempt = 0; attempt < params.tries; ++attempt) {
        std::vector<Point2D> controlPoints = sampleControlPoints(data, indices, params.numberOfControlPoints, rng);

        if (!satisfiesSpacing(controlPoints, params.minControlPointXGap, params.maxControlPointYGap)) {
            continue;
        }

        const std::vector<Point2D> candidateCurve = buildCandidateCurve(controlPoints, params, dataXMin, dataXMax);
        auto [inlierMask, inlierCount] = scoreCandidate(data, controlPoints, candidateCurve, params);

        if (inlierCount > best.inlierCount) {
            best.controlPoints = std::move(controlPoints);
            best.inlierMask = std::move(inlierMask);
            best.inlierCount = inlierCount;
        }
    }

    if (best.inlierCount > 0) {
        const auto [extendXMin, extendXMax] = getDataRange(data, best, params, dataXMin, dataXMax);
        best.curve = getCardinalSplineCurve(best.controlPoints, params.tension, params.samplesPerSegment);
        best.curve = extendSplineEndsLinearly(best.curve, extendXMin, extendXMax);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "[fitRansac] %.1f ms (%d/%d inliers)\n", elapsedMs, best.inlierCount, (int)data.size());

    return best;
}

}  // namespace rsf
