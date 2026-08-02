#pragma once

#include <cmath>
#include <vector>

namespace rsf {

struct Point2D { // Cpp: Point2D p = {3.0, 4.0};  to use
    double x = 0.0;
    double y = 0.0;

    Point2D operator+(const Point2D& o) const { return {x + o.x, y + o.y}; } // Cpp: operator overloading
    Point2D operator-(const Point2D& o) const { return {x - o.x, y - o.y}; }
    Point2D operator*(double s) const { return {x * s, y * s}; }
};

inline double distance(const Point2D& a, const Point2D& b) { // Cpp: inline because it is in .hpp, so it is pasted in many .cpp that use it, otherwise we would get the error multiple definitions
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Evaluates one segment of a cardinal spline between p1 and p2, using the
// neighboring control points p0 and p3 to derive tangents. t in [0, 1].
// tension in [0, 1]: 0 gives a Catmull-Rom spline, 1 flattens tangents to zero.
inline Point2D evalCardinalSegment(const Point2D& p0, const Point2D& p1,
                                    const Point2D& p2, const Point2D& p3,
                                    double t, double tension) {
                                        
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

// Samples a full cardinal spline through an ordered list of control points.
// Phantom points are synthesized at both ends so the curve passes through
// every control point, including the first and last.
// Requires at least 2 control points.
inline std::vector<Point2D> sampleCardinalSpline(const std::vector<Point2D>& controlPoints,
                                                   double tension,
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

}  // namespace rsf
