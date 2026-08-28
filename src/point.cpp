#include "rsf/point.hpp"

#include <algorithm>
#include <cmath>

namespace rsf {

double distance(const Point2D& a, const Point2D& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double dot(const Point2D& a, const Point2D& b) { return a.x * b.x + a.y * b.y; }

std::pair<double, double> dataXRange(const std::vector<Point2D>& data) {
    double xMin = data[0].x;
    double xMax = data[0].x;
    for (const auto& p : data) {
        xMin = std::min(xMin, p.x);
        xMax = std::max(xMax, p.x);
    }
    return {xMin, xMax};
}

}  // namespace rsf
