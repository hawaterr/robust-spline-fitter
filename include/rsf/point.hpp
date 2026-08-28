#pragma once

#include <utility>
#include <vector>

namespace rsf {

struct Point2D {  // Cpp: Point2D p = {3.0, 4.0};  to use
    double x = 0.0;
    double y = 0.0;

    Point2D operator+(const Point2D& o) const { return {x + o.x, y + o.y}; }  // Cpp: operator overloading
    Point2D operator-(const Point2D& o) const { return {x - o.x, y - o.y}; }
    Point2D operator*(double s) const { return {x * s, y * s}; }
};

double distance(const Point2D& a, const Point2D& b);

double dot(const Point2D& a, const Point2D& b);

// Min/max x over data, used to extend a candidate curve's linear ends to
// cover the full data range regardless of where its control points landed.
std::pair<double, double> dataXRange(const std::vector<Point2D>& data);

}  // namespace rsf
