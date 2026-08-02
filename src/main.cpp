#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rsf/cardinal_spline.hpp"

using rsf::Point2D; // Cpp: so we can do Point2D directly

namespace {

// Loads points from a CSV with an "x,y" header
std::vector<Point2D> loadCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("could not open " + path);
    }

    std::vector<Point2D> points;
    std::string line;
    std::getline(in, line);  // header

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string xStr, yStr;
        std::getline(ss, xStr, ',');
        std::getline(ss, yStr, ',');
        points.push_back({std::stod(xStr), std::stod(yStr)});
    }

    return points;
}

// very expensive, could maybe optimize here?
double distanceToCurve(const Point2D& point, const std::vector<Point2D>& curve) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& c : curve) {
        best = std::min(best, rsf::distance(point, c));
    }
    return best;
}

// sortedControlPoints must already be sorted by x.
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

struct FitResult {
    std::vector<Point2D> controlPoints;
    std::vector<Point2D> curve;
    std::vector<bool> inlierMask;
    int inlierCount = 0;
};

FitResult fitRansac(const std::vector<Point2D>& data, int numberOfControlPoints, int tries, double tension,
                     double threshold, int samplesPerSegment, double minControlPointXGap,
                     double maxControlPointYGap, std::mt19937& rng) {
    FitResult best;

    double dataXMin = data[0].x;
    double dataXMax = data[0].x;
    for (const auto& p : data) {
        dataXMin = std::min(dataXMin, p.x);
        dataXMax = std::max(dataXMax, p.x);
    }

    std::vector<int> indices(data.size()); //
    std::iota(indices.begin(), indices.end(), 0); // Cpp: fills indices with [0,1,2,...], .begin() returns an iterator

    for (int attempt = 0; attempt < tries; ++attempt) {
        std::vector<int> sample = indices;
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(numberOfControlPoints);
        // Order control points along the curve by x. This assumes the curve is a
        // function of x (single y per x), which holds for this synthetic test data
        // but won't hold in general (e.g. closed loops or non-monotonic curves).
        std::sort(sample.begin(), sample.end(),
                  [&](int a, int b) { return data[a].x < data[b].x; });

        std::vector<Point2D> controlPoints;
        controlPoints.reserve(numberOfControlPoints);
        for (int idx : sample) controlPoints.push_back(data[idx]);

        if (!satisfiesSpacing(controlPoints, minControlPointXGap, maxControlPointYGap)) {
            continue;
        }

        std::vector<Point2D> curve = rsf::sampleCardinalSpline(controlPoints, tension, samplesPerSegment);
        curve = rsf::extendSplineLinear(curve, dataXMin, dataXMax);

        std::vector<bool> inlierMask(data.size());
        int inlierCount = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            const bool isInlier = distanceToCurve(data[i], curve) < threshold;
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

void writeCsv(const std::string& path, const std::vector<Point2D>& points,
              const std::vector<bool>* inlierMask = nullptr) {
    std::ofstream out(path);
    out << "x,y" << (inlierMask ? ",inlier" : "") << "\n";
    for (size_t i = 0; i < points.size(); ++i) {
        out << points[i].x << "," << points[i].y;
        if (inlierMask) out << "," << ((*inlierMask)[i] ? 1 : 0);
        out << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) { // Cpp: argc/argv: argc is the count of command-line arguments; argv is the array of those arguments as C-strings. argv[0] is the program name, argv[1] onward are user-supplied args.
    const std::string inputPath = "../data/data.csv"; // Cpp: const: marks a variable as immutable after initialization — the compiler errors if you try to modify it later.
    const int numberOfControlPoints = 4;
    const int tries = 50000;
    const double threshold = 0.2;
    const double tension = 0.5;
    const int samplesPerSegment = 200;
    const double minControlPointXGap = 1.0;
    const double maxControlPointYGap = 2.0;

    std::mt19937 rng(42);

    const std::vector<Point2D> data = loadCsv(inputPath);
    const FitResult best = fitRansac(data, numberOfControlPoints, tries, tension, threshold, samplesPerSegment,
                                      minControlPointXGap, maxControlPointYGap, rng);

    std::cout << "Loaded " << data.size() << " points from " << inputPath << "\n";

    if (best.controlPoints.empty()) {
        std::cout << "No candidate spline found: every sampled control-point set was rejected "
                     "(check minControlPointXGap/maxControlPointYGap against the data's scale).\n";
        return 1;
    }

    std::cout << "Best fit: " << best.inlierCount << " / " << data.size() << " inliers using "
              << numberOfControlPoints << " control points over " << tries << " tries\n";

    writeCsv("data_points.csv", data, &best.inlierMask);
    writeCsv("spline_curve.csv", best.curve);
    writeCsv("control_points.csv", best.controlPoints);

    std::cout << "Wrote data_points.csv, spline_curve.csv, control_points.csv\n";
    return 0;
}
