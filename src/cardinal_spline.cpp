#include "rsf/cardinal_spline.hpp"

#include <cmath>

namespace rsf {

double distance(const Point2D& a, const Point2D& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

Point2D evalCardinalSegment(const Point2D& p0, const Point2D& p1, const Point2D& p2, const Point2D& p3, double t,
                             double tension) {
    const double s = 1.0 - tension;
    const Point2D m1 = (p2 - p0) * (s * 0.5);
    const Point2D m2 = (p3 - p1) * (s * 0.5);

    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2 * t3 - 3 * t2 + 1;
    const double h10 = t3 - 2 * t2 + t;
    const double h01 = -2 * t3 + 3 * t2;
    const double h11 = t3 - t2;

    return p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11;
}

std::vector<Point2D> getCardinalSplineCurve(const std::vector<Point2D>& controlPoints, double tension,
                                           int samplesPerSegment) {
    std::vector<Point2D> curve;
    const int n = static_cast<int>(controlPoints.size()); // Cpp: static_cast<int>(x) converts x to an int, .size is size_t an unsigned integer, but since we do operations like n-2, we make sure no compiler errors or warnings
    if (n < 2 || samplesPerSegment < 1) {
        return curve;
    }

    auto selectControlPoint = [&](int i) -> Point2D { //Cpp: selectControlPoint is a variable storing a function, the value is the output of lambda function (which does ), [] is called capture, and it is what the function can see from outside, [&] means it can see everything by reference
        if (i < 0) return controlPoints[0] * 2.0 - controlPoints[1];
        if (i >= n) return controlPoints[n - 1] * 2.0 - controlPoints[n - 2];
        return controlPoints[i];
    };

    curve.reserve((n - 1) * samplesPerSegment + 1); // Cpp: no need to keep pushing and reallocating
    for (int i = 0; i < n - 1; ++i) {
        const Point2D& p0 = selectControlPoint(i - 1);
        const Point2D& p1 = selectControlPoint(i);
        const Point2D& p2 = selectControlPoint(i + 1);
        const Point2D& p3 = selectControlPoint(i + 2);
        const int lastSample = (i == n - 2) ? samplesPerSegment : samplesPerSegment - 1; // Cpp: ternary expression condition ? value_if_true : value_if_false
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
