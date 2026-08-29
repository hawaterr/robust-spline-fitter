#include "rsf/distance_to_spline_newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "rsf/point.hpp"

namespace rsf {

namespace {

double singleNewtonRun(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3, const Point2D& query,
                       double tension, double tStart) {
    constexpr int maxIters = 8;
    constexpr double tol = 1e-6;

    double t = tStart;
    for (int iter = 0; iter < maxIters; ++iter) {
        const Point2D P = evalCardinalSegment(p0, p1, p2, p3, t, tension);
        const Point2D Pd = evalCardinalSegmentDerivative(p0, p1, p2, p3, t, tension);
        const Point2D Pdd = evalCardinalSegmentSecondDerivative(p0, p1, p2, p3, t, tension);
        const Point2D diff = P - query;

        const double numerator = dot(diff, Pd);
        const double denominator = dot(Pd, Pd) + dot(diff, Pdd);
        if (std::abs(denominator) < 1e-12) {
            break;
        }

        double tNext = t - numerator / denominator;
        tNext = std::max(0.0, std::min(1.0, tNext));
        const bool converged = std::abs(tNext - t) < tol;
        t = tNext;
        if (converged) {
            break;
        }
    }
    return t;
}

// projection from linar algebra course
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

}  // namespace

double getClosestSquaredDistanceToSegmentUsingNewton(const Point2D& p0, const Point2D& p1, const Point2D& p2,
                                                     const Point2D& p3, const Point2D& query, double tension,
                                                     double* outT) {
    double proportionalGuess = 0.5;
    if (p2.x != p1.x) {
        proportionalGuess = (query.x - p1.x) / (p2.x - p1.x);
        proportionalGuess = std::max(0.0, std::min(1.0, proportionalGuess));
    }

    // A single Newton seed can converge to the wrong local minimum or get
    // stuck near a boundary when the initial guess is far from the true
    // closest point (e.g. a query far off the segment's x-range). Running
    // from several seeds and keeping the global best is cheap (a handful of
    // Newton solves, still O(1) per query) and much more robust.
    double bestDistSq = std::numeric_limits<double>::infinity();
    double bestT = 0.0;
    for (double seed : {0.0, 0.25, 0.5, 0.75, 1.0, proportionalGuess}) {
        const double t = singleNewtonRun(p0, p1, p2, p3, query, tension, seed);
        const Point2D P = evalCardinalSegment(p0, p1, p2, p3, t, tension);
        const Point2D diff = P - query;
        const double distSq = dot(diff, diff);
        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            bestT = t;
        }
    }

    if (outT != nullptr) {
        *outT = bestT;
    }
    return bestDistSq;
}

double getClosestDistanceToSplineUsingNewton(const Point2D& point, const std::vector<Point2D>& sortedControlPoints,
                                             double tension) {
    const int n = static_cast<int>(sortedControlPoints.size());

    // Check every segment
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n - 1; ++i) {
        const Point2D p0 = selectControlPoint(sortedControlPoints, i - 1);
        const Point2D& p1 = sortedControlPoints[i];
        const Point2D& p2 = sortedControlPoints[i + 1];
        const Point2D p3 = selectControlPoint(sortedControlPoints, i + 2);
        const double distSq = getClosestSquaredDistanceToSegmentUsingNewton(p0, p1, p2, p3, point, tension);
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

}  // namespace rsf
