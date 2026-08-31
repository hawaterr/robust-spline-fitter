#include "rsf/cardinal_spline.hpp"

#include <algorithm>
#include <cmath>

namespace rsf {

Point2D selectControlPoint(const std::vector<Point2D>& controlPoints, int i) {
    const int n = static_cast<int>(controlPoints.size());
    if (i < 0) return controlPoints[0] * 2.0 - controlPoints[1]; // mirrors the control point at the ends
    if (i >= n) return controlPoints[n - 1] * 2.0 - controlPoints[n - 2];
    return controlPoints[i];
}

Point2D evalCardinalSegment(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3, double t,
                            double tension) {
    const double s = 1.0 - tension;
    const Point2D m1 = (p2 - p0) * (s * 0.5);
    const Point2D m2 = (p3 - p1) * (s * 0.5);

    const double t2 = t * t;
    const double t3 = t2 * t;
    // hermite spline basis, where m1 and m2 (or v1 and v2) are calculate automatically
    const double h00 = 2 * t3 - 3 * t2 + 1;
    const double h10 = t3 - 2 * t2 + t;
    const double h01 = -2 * t3 + 3 * t2;
    const double h11 = t3 - t2;

    return p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11;
}

Point2D evalCardinalSegmentDerivative(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3,
                                      double t, double tension) {
    const double s = 1.0 - tension;
    const Point2D m1 = (p2 - p0) * (s * 0.5);
    const Point2D m2 = (p3 - p1) * (s * 0.5);

    const double t2 = t * t;
    const double h00d = 6 * t2 - 6 * t;
    const double h10d = 3 * t2 - 4 * t + 1;
    const double h01d = -6 * t2 + 6 * t;
    const double h11d = 3 * t2 - 2 * t;

    return p1 * h00d + m1 * h10d + p2 * h01d + m2 * h11d;
}

Point2D evalCardinalSegmentSecondDerivative(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3,
                                            double t, double tension) {
    const double s = 1.0 - tension;
    const Point2D m1 = (p2 - p0) * (s * 0.5);
    const Point2D m2 = (p3 - p1) * (s * 0.5);

    const double h00dd = 12 * t - 6;
    const double h10dd = 6 * t - 4;
    const double h01dd = -12 * t + 6;
    const double h11dd = 6 * t - 2;

    return p1 * h00dd + m1 * h10dd + p2 * h01dd + m2 * h11dd;
}

// control points come in already in the right order the user wants
std::vector<Point2D> getCardinalSplineCurve(const std::vector<Point2D>& controlPoints, double tension,
                                            int samplesPerSegment) {
    std::vector<Point2D> curve;
    const int n = static_cast<int>(
        controlPoints.size());  // Cpp: static_cast<int>(x) converts x to an int, .size is size_t an unsigned integer,
                                // but since we do operations like n-2, we make sure no compiler errors or warnings
    if (n < 2 || samplesPerSegment < 1) {
        return curve;
    }

    curve.reserve((n - 1) * samplesPerSegment + 1);  // Cpp: no need to keep pushing and reallocating
    for (int i = 0; i < n - 1; ++i) {
        const Point2D p0 = selectControlPoint(controlPoints, i - 1);
        const Point2D p1 = selectControlPoint(controlPoints, i);
        const Point2D p2 = selectControlPoint(controlPoints, i + 1);
        const Point2D p3 = selectControlPoint(controlPoints, i + 2);
        const int lastSample =
            (i == n - 2) ? samplesPerSegment
                         : samplesPerSegment - 1;  // Cpp: ternary expression condition ? value_if_true : value_if_false
        for (int s = 0; s <= lastSample; ++s) {
            const double t = static_cast<double>(s) / samplesPerSegment;
            curve.push_back(evalCardinalSegment(p0, p1, p2, p3, t, tension));
        }
    }
    return curve;
}

Point2D evalCardinalSplineAtU(const std::vector<Point2D>& controlPoints, double u, double tension) {
    const int n = static_cast<int>(controlPoints.size());
    if (n < 2) {
        return n == 1 ? controlPoints[0] : Point2D{};
    }

    const double clampedU = std::max(0.0, std::min(static_cast<double>(n - 1), u));
    // The last segment owns u == n-1 (at t == 1), so clamp the index to n-2.
    const int i = std::min(static_cast<int>(std::floor(clampedU)), n - 2);
    const double t = clampedU - i;

    const Point2D p0 = selectControlPoint(controlPoints, i - 1);
    const Point2D p1 = selectControlPoint(controlPoints, i);
    const Point2D p2 = selectControlPoint(controlPoints, i + 1);
    const Point2D p3 = selectControlPoint(controlPoints, i + 2);
    return evalCardinalSegment(p0, p1, p2, p3, t, tension);
}

std::vector<Point2D> extendSplineEndsLinearly(const std::vector<Point2D>& curve, double xMin, double xMax,
                                              int samplesPerExtension) {
    if (curve.size() < 2) {
        return curve;
    }

    const Point2D& first = curve.front();
    const Point2D& second = curve[1];
    const Point2D& last = curve.back();
    const Point2D& beforeLast = curve[curve.size() - 2];

    const double slopeStart = (second.y - first.y) / (second.x - first.x);
    const double slopeEnd = (last.y - beforeLast.y) / (last.x - beforeLast.x);

    std::vector<Point2D> extended;
    extended.reserve(curve.size() + 2 * samplesPerExtension);

    if (xMin < first.x) {
        for (int i = 0; i < samplesPerExtension; ++i) {
            const double t = static_cast<double>(i) / (samplesPerExtension - 1);
            const double x = xMin + t * (first.x - xMin);
            extended.push_back({x, first.y + (x - first.x) * slopeStart});
        }
    }

    extended.insert(extended.end(), curve.begin(), curve.end());

    if (xMax > last.x) {
        for (int i = 0; i < samplesPerExtension; ++i) {
            const double t = static_cast<double>(i) / (samplesPerExtension - 1);
            const double x = last.x + t * (xMax - last.x);
            extended.push_back({x, last.y + (x - last.x) * slopeEnd});
        }
    }

    return extended;
}

// extends till the last inlier
std::vector<Point2D> extendSplineEndsAlongTangent(const std::vector<Point2D>& curve,
                                                  const std::vector<Point2D>& controlPoints,
                                                  const std::vector<Point2D>& inliers, double tension,
                                                  int samplesPerExtension) {
    const int n = static_cast<int>(controlPoints.size());
    if (curve.size() < 2 || n < 2 || inliers.empty()) {
        return curve;
    }

    const Point2D startTangent =
        evalCardinalSegmentDerivative(selectControlPoint(controlPoints, -1), controlPoints[0], controlPoints[1],
                                      selectControlPoint(controlPoints, 2), 0.0, tension) *
        -1.0;
    const Point2D endTangent =
        evalCardinalSegmentDerivative(selectControlPoint(controlPoints, n - 3), controlPoints[n - 2],
                                      controlPoints[n - 1], selectControlPoint(controlPoints, n), 1.0, tension);

    const auto travel = [&](const Point2D& from, const Point2D& dir) { // how far do inliers reach in this direction
        const double length = std::hypot(dir.x, dir.y);
        if (length <= 0.0) {
            return 0.0;
        }
        const Point2D unit = {dir.x / length, dir.y / length};
        double furthest = 0.0;
        for (const Point2D& p : inliers) {
            furthest = std::max(furthest, (p.x - from.x) * unit.x + (p.y - from.y) * unit.y);
        }
        return furthest;
    };

    std::vector<Point2D> extended;
    extended.reserve(curve.size() + 2 * samplesPerExtension);

    const Point2D& first = curve.front();
    const Point2D& last = curve.back();
    const double startLength = travel(first, startTangent);
    const double endLength = travel(last, endTangent);

    if (startLength > 0.0) {
        const double norm = std::hypot(startTangent.x, startTangent.y);
        for (int i = samplesPerExtension - 1; i >= 0; --i) {
            const double s = startLength * static_cast<double>(i) / (samplesPerExtension - 1);
            extended.push_back({first.x + startTangent.x / norm * s, first.y + startTangent.y / norm * s});
        }
    }

    extended.insert(extended.end(), curve.begin(), curve.end());

    if (endLength > 0.0) {
        const double norm = std::hypot(endTangent.x, endTangent.y);
        for (int i = 0; i < samplesPerExtension; ++i) {
            const double s = endLength * static_cast<double>(i) / (samplesPerExtension - 1);
            extended.push_back({last.x + endTangent.x / norm * s, last.y + endTangent.y / norm * s});
        }
    }

    return extended;
}

}  // namespace rsf
