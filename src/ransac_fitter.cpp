#include "rsf/ransac_fitter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "rsf/distance_to_spline_newton.hpp"
#include "rsf/distance_to_spline_other.hpp"
#include "rsf/point.hpp"
#include "rsf/point_ordering.hpp"

namespace rsf {

namespace {


constexpr int kMaxAttemptsPerTry = 100;

// Draws numberOfControlPoints distinct indices from `indices` (via partial
// Fisher-Yates on a copy) and returns the corresponding data points ordered
// along the curve per `curveType`, ready to serve as spline control points.
std::vector<Point2D> sampleControlPoints(const std::vector<Point2D>& data, std::vector<int> indices,
                                         int numberOfControlPoints, CurveType curveType, std::mt19937& rng) {
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(numberOfControlPoints);
    if (curveType == CurveType::Explicit) {
        // Ordering by x assumes the curve is a function of x (single y per x).
        // Use CurveType::Implicit for curves that fold back.
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return data[a].x < data[b].x; });
    }

    std::vector<Point2D> controlPoints;
    controlPoints.reserve(numberOfControlPoints);
    for (int idx : indices) controlPoints.push_back(data[idx]);

    if (curveType == CurveType::Implicit) {
        orderPointsByProximity(controlPoints);
    }
    return controlPoints;
}

std::vector<Point2D> buildCandidateCurve(const std::vector<Point2D>& data, const std::vector<Point2D>& controlPoints,
                                         const RansacFitParams& params, double dataXMin, double dataXMax) {
    if (params.distanceMetric == DistanceMetric::Newton) {
        return {};  // no need to build the whole curve if we use the newton method (optimisation)
    }
    std::vector<Point2D> curve = getCardinalSplineCurve(controlPoints, params.tension, params.samplesPerSegment);

    if (params.curveType == CurveType::Implicit) {
        return extendSplineEndsAlongTangent(curve, controlPoints, data, params.tension);
    } else {
        return extendSplineEndsLinearly(curve, dataXMin, dataXMax);
    }
}

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

std::vector<Point2D> collectInliers(const std::vector<Point2D>& data, const FitResult& best) {
    std::vector<Point2D> inliers;
    inliers.reserve(best.inlierCount);
    for (size_t i = 0; i < data.size(); ++i) {
        if (best.inlierMask[i]) inliers.push_back(data[i]);
    }
    return inliers;
}

// Picks the x-range the final curve is extended to, in case the user wants to
// only see the useful part of the best winning curve
std::pair<double, double> getDataRange(const std::vector<Point2D>& data, const FitResult& best,
                                       const RansacFitParams& params, double dataXMin, double dataXMax) {
    if (params.fitRange == FitRange::DataXRange) {
        return {dataXMin, dataXMax};
    } else {
        return dataXRange(collectInliers(data, best));
    }
}

}  // namespace

bool satisfiesSpacing(const std::vector<Point2D>& controlPoints, double minXGap, double maxYGap, CurveType curveType) {
    for (size_t i = 0; i + 1 < controlPoints.size(); ++i) {
        if (curveType == CurveType::Implicit) {
            // if (distance(controlPoints[i], controlPoints[i + 1]) <= minD) {
            //     return false;
            // } TODO: add
            return true;
        }
        const double xGap = controlPoints[i + 1].x - controlPoints[i].x;
        const double yGap = std::abs(controlPoints[i + 1].y - controlPoints[i].y);
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


    const long long maxAttempts = static_cast<long long>(params.tries) * kMaxAttemptsPerTry;
    int scored = 0;
    long long attempts = 0;
    while (scored < params.tries && attempts < maxAttempts) { // number of tries only counts when we have actually tried and not skiped due to bad control points
        ++attempts;
        std::vector<Point2D> controlPoints =
            sampleControlPoints(data, indices, params.numberOfControlPoints, params.curveType, rng);

        if (!satisfiesSpacing(controlPoints, params.minControlPointXGap, params.maxControlPointYGap,
                              params.curveType)) {
            continue;
        }
        ++scored;

        const std::vector<Point2D> candidateCurve =
            buildCandidateCurve(data, controlPoints, params, dataXMin, dataXMax);
        auto [inlierMask, inlierCount] = scoreCandidate(data, controlPoints, candidateCurve, params);

        if (inlierCount > best.inlierCount) {
            best.controlPoints = std::move(controlPoints);
            best.inlierMask = std::move(inlierMask);
            best.inlierCount = inlierCount;
        }
    }

    if (scored < params.tries) {
        std::fprintf(stderr,
                     "[fitRansac] warning: only %d/%d samples met the spacing constraints in %lld attempts; "
                     "minControlPointXGap/maxControlPointYGap may be too tight for this data\n",
                     scored, params.tries, attempts);
    }

    if (best.inlierCount > 0) {
        // rebuild curve since user might only want till the last inlier
        best.curve = getCardinalSplineCurve(best.controlPoints, params.tension, params.samplesPerSegment);
        if (params.curveType == CurveType::Implicit) {
            best.curve = extendSplineEndsAlongTangent(best.curve, best.controlPoints, collectInliers(data, best),
                                                      params.tension);
        } else {
            const auto [extendXMin, extendXMax] = getDataRange(data, best, params, dataXMin, dataXMax);
            best.curve = extendSplineEndsLinearly(best.curve, extendXMin, extendXMax);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "[fitRansac] %.1f ms (%d/%d inliers)\n", elapsedMs, best.inlierCount, (int)data.size());

    // std::fprintf(stderr, "[fitRansac] control points:");
    // for (const auto& p : best.controlPoints) {
    //     std::fprintf(stderr, " (%.4f, %.4f)", p.x, p.y);
    // }
    // std::fprintf(stderr, "\n");

    return best;
}

}  // namespace rsf
