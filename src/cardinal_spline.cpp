#include "rsf/cardinal_spline.hpp"

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

}  // namespace rsf
