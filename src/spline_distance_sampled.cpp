#include "rsf/spline_distance_sampled.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rsf {

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

}  // namespace rsf
