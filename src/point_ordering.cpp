#include "rsf/point_ordering.hpp"

#include <algorithm>
#include <limits>

namespace rsf {

// uses DP to find shortest path (Held Karp algorithm, well known algo)
void orderPointsByProximity(std::vector<Point2D>& points) {
    const int n = static_cast<int>(points.size());
    if (n < 3) {
        return;  // 0, 1 or 2 points: only one path exists (up to direction)
    }

    const double kInfinity = std::numeric_limits<double>::infinity();
    const int subsetCount = 1 << n;  // Cpp: 1 << n is 2^n, one bit per point

    // Cpp: thread_local statics reused across calls so the RANSAC loop does no
    // per-try heap allocation. assign() refills them without reallocating once
    // they've grown to size.
    static thread_local std::vector<double> distances;
    static thread_local std::vector<double> shortestPathLength;
    static thread_local std::vector<int> previousPoint;

    distances.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double d = distance(points[i], points[j]);
            distances[i * n + j] = d;
            distances[j * n + i] = d;
        }
    }

    // Held-Karp: shortestPathLength[subset][last] is the length of the shortest
    // path that visits exactly the points in `subset` and ends at `last`.
    shortestPathLength.assign(subsetCount * n, kInfinity);
    previousPoint.assign(subsetCount * n, -1);

    // An open path may start anywhere, so every single-point subset costs 0.
    for (int i = 0; i < n; ++i) {
        shortestPathLength[(1 << i) * n + i] = 0.0;
    }

    for (int subset = 1; subset < subsetCount; ++subset) {
        for (int last = 0; last < n; ++last) {
            const double lengthSoFar = shortestPathLength[subset * n + last];
            if (!(subset >> last & 1) || lengthSoFar == kInfinity) {
                continue;  // `last` isn't in this subset, or the state is unreachable
            }
            for (int next = 0; next < n; ++next) {
                if (subset >> next & 1) {
                    continue;  // already visited
                }
                const int grownSubset = subset | 1 << next;
                const double candidate = lengthSoFar + distances[last * n + next];
                if (candidate < shortestPathLength[grownSubset * n + next]) {
                    shortestPathLength[grownSubset * n + next] = candidate;
                    previousPoint[grownSubset * n + next] = last;
                }
            }
        }
    }

    // The best full path is the cheapest way to end up having visited everything.
    const int allPoints = subsetCount - 1;
    int endPoint = 0;
    double bestLength = kInfinity;
    for (int i = 0; i < n; ++i) {
        if (shortestPathLength[allPoints * n + i] < bestLength) {
            bestLength = shortestPathLength[allPoints * n + i];
            endPoint = i;
        }
    }

    // Walk the parent pointers back from the end to recover the order.
    std::vector<Point2D> ordered;
    ordered.reserve(n);
    for (int subset = allPoints, at = endPoint; at != -1;) {
        ordered.push_back(points[at]);
        const int previous = previousPoint[subset * n + at];
        subset ^= 1 << at;
        at = previous;
    }
    std::reverse(ordered.begin(), ordered.end());

    // Both directions of an open path have the same length, so pick one
    // deterministically: increasing x, tie-broken on y. This also keeps the
    // output consistent with x-sorting on data that is a function of x.
    const Point2D& first = ordered.front();
    const Point2D& last = ordered.back();
    if (last.x < first.x || (last.x == first.x && last.y < first.y)) {
        std::reverse(ordered.begin(), ordered.end());
    }

    points = std::move(ordered);
}

}  // namespace rsf
