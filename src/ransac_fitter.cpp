#include "rsf/ransac_fitter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace rsf {

double distanceToCurve(const Point2D& point, const std::vector<Point2D>& curve) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& c : curve) {
        best = std::min(best, distance(point, c));
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
    FitResult best;

    double dataXMin = data[0].x;
    double dataXMax = data[0].x;
    for (const auto& p : data) {
        dataXMin = std::min(dataXMin, p.x);
        dataXMax = std::max(dataXMax, p.x);
    }

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0); // Cpp: fills indices with [0,1,2,...], .begin() returns an iterator

    for (int attempt = 0; attempt < params.tries; ++attempt) {
        std::vector<int> sample = indices;
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(params.numberOfControlPoints);
        // Order control points along the curve by x. This assumes the curve is a
        // function of x (single y per x), which holds for this synthetic test data
        // but won't hold in general (e.g. closed loops or non-monotonic curves).
        std::sort(sample.begin(), sample.end(), [&](int a, int b) { return data[a].x < data[b].x; });

        std::vector<Point2D> controlPoints;
        controlPoints.reserve(params.numberOfControlPoints);
        for (int idx : sample) controlPoints.push_back(data[idx]);

        if (!satisfiesSpacing(controlPoints, params.minControlPointXGap, params.maxControlPointYGap)) {
            continue;
        }

        std::vector<Point2D> curve = getCardinalSplineCurve(controlPoints, params.tension, params.samplesPerSegment);
        curve = extendSplineEndsLinearly(curve, dataXMin, dataXMax);

        std::vector<bool> inlierMask(data.size());
        int inlierCount = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            const bool isInlier = distanceToCurve(data[i], curve) < params.threshold;
            inlierMask[i] = isInlier;
            if (isInlier) ++inlierCount;
        }

        if (inlierCount > best.inlierCount) {
            best.controlPoints = std::move(controlPoints);
            best.curve = std::move(curve);
            best.inlierMask = std::move(inlierMask);
            best.inlierCount = inlierCount;
        }
    }

    return best;
}

}  // namespace rsf
