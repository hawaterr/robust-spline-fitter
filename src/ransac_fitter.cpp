#include "rsf/ransac_fitter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace rsf {

namespace {

// Mirrors getCardinalSplineCurve's selectControlPoint: phantom points are
// synthesized by linear reflection at both ends so boundary segments still
// have a well-defined p0/p3.
Point2D selectControlPoint(const std::vector<Point2D>& controlPoints, int i) {
    const int n = static_cast<int>(controlPoints.size());
    if (i < 0) return controlPoints[0] * 2.0 - controlPoints[1];
    if (i >= n) return controlPoints[n - 1] * 2.0 - controlPoints[n - 2];
    return controlPoints[i];
}

// Closed-form squared distance from `query` to the ray starting at `origin`
// in direction `dir`, clamped so the projection cannot fall behind `origin`
// (one-sided extension, matching extendSplineEndsLinearly).
double rayDistanceSquared(const Point2D& origin, const Point2D& dir, const Point2D& query) {
    const Point2D originToQuery = query - origin;
    const double dirDot = dir.x * dir.x + dir.y * dir.y;
    if (dirDot < 1e-12) {
        const Point2D diff = query - origin;
        return diff.x * diff.x + diff.y * diff.y;
    }
    double proj = (originToQuery.x * dir.x + originToQuery.y * dir.y) / dirDot;
    proj = std::max(0.0, proj);
    const Point2D closest = origin + dir * proj;
    const Point2D diff = closest - query;
    return diff.x * diff.x + diff.y * diff.y;
}

// Min/max x over data, used to extend the candidate curve's linear ends to
// cover the full data range regardless of where its control points landed.
std::pair<double, double> dataXRange(const std::vector<Point2D>& data) {
    double xMin = data[0].x;
    double xMax = data[0].x;
    for (const auto& p : data) {
        xMin = std::min(xMin, p.x);
        xMax = std::max(xMax, p.x);
    }
    return {xMin, xMax};
}



// Builds the densely-sampled, linearly-extended curve needed by the
// Vertical/Sampled metrics. Perpendicular scores directly off control points
// and never calls this, so it's skipped there to avoid the sampling cost.
std::vector<Point2D> buildCandidateCurve(const std::vector<Point2D>& controlPoints, const RansacFitParams& params,
                                          double dataXMin, double dataXMax) {
    if (params.distanceMetric != DistanceMetric::Vertical && params.distanceMetric != DistanceMetric::Sampled) {
        return {};
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
                dist = distanceToSpline(data[i], controlPoints, params.tension);
                break;
        }
        const bool isInlier = dist < params.threshold;
        inlierMask[i] = isInlier;
        if (isInlier) ++inlierCount;
    }
    return {std::move(inlierMask), inlierCount};
}

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

}  // namespace

double distanceToSpline(const Point2D& point, const std::vector<Point2D>& sortedControlPoints, double tension) {
    const int n = static_cast<int>(sortedControlPoints.size());

    // Check every segment
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n - 1; ++i) {
        const Point2D p0 = selectControlPoint(sortedControlPoints, i - 1);
        const Point2D& p1 = sortedControlPoints[i];
        const Point2D& p2 = sortedControlPoints[i + 1];
        const Point2D p3 = selectControlPoint(sortedControlPoints, i + 2);
        const double distSq = closestDistanceSquaredOnSegment(p0, p1, p2, p3, point, tension);
        best = std::min(best, distSq);
    }

    // The two linear extension segments
    {
        const Point2D p0 = selectControlPoint(sortedControlPoints, -1);
        const Point2D& p1 = sortedControlPoints[0];
        const Point2D& p2 = sortedControlPoints[1];
        const Point2D p3 = selectControlPoint(sortedControlPoints, 2);
        const Point2D dir = evalCardinalSegmentDerivative(p0, p1, p2, p3, 0.0, tension);
        best = std::min(best, rayDistanceSquared(p1, dir * -1.0, point));
    }
    {
        const Point2D p0 = selectControlPoint(sortedControlPoints, n - 3);
        const Point2D& p1 = sortedControlPoints[n - 2];
        const Point2D& p2 = sortedControlPoints[n - 1];
        const Point2D p3 = selectControlPoint(sortedControlPoints, n);
        const Point2D dir = evalCardinalSegmentDerivative(p0, p1, p2, p3, 1.0, tension);
        best = std::min(best, rayDistanceSquared(p2, dir, point));
    }

    return std::sqrt(best);
}

double verticalDistanceToSpline(const Point2D& point, const std::vector<Point2D>& curveSamples) {
    
    // std::upper_bound does a binary search assuming curveSamples is sorted by x. It returns an iterator to the first sample whose x is strictly greater than point.x.
    const auto it = std::upper_bound(curveSamples.begin(), curveSamples.end(), point.x,
                                      [](double x, const Point2D& p) { return x < p.x; }); 

    // no need to interpolate
    if (it == curveSamples.begin()) {
        return std::abs(curveSamples.front().y - point.y);
    }
    if (it == curveSamples.end()) {
        return std::abs(curveSamples.back().y - point.y);
    }

    // interpolates the y value between the 2 discretized curve points
    const Point2D& after = *it;
    const Point2D& before = *(it - 1);
    const double span = after.x - before.x;
    const double curveY = (span < 1e-12) ? before.y : before.y + (after.y - before.y) * (point.x - before.x) / span;
    return std::abs(curveY - point.y);
}

double sampledDistanceToSpline(const Point2D& point, const std::vector<Point2D>& curveSamples) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& sample : curveSamples) {
        best = std::min(best, distance(point, sample));
    }
    return best;
}

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
    std::iota(indices.begin(), indices.end(), 0); // Cpp: fills indices with [0,1,2,...], .begin() returns an iterator

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
        best.curve = getCardinalSplineCurve(best.controlPoints, params.tension, params.samplesPerSegment);
        best.curve = extendSplineEndsLinearly(best.curve, dataXMin, dataXMax);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "[fitRansac] %.1f ms (%d/%d inliers)\n", elapsedMs, best.inlierCount, (int)data.size());

    return best;
}

}  // namespace rsf
