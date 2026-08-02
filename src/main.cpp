#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "rsf/cardinal_spline.hpp"

using rsf::Point2D; // Cpp: so we can do Point2D directly

namespace {

// Ground-truth curve used to generate synthetic test data: a sine wave.
Point2D trueCurve(double x) {
    return {x, std::sin(x * 1.5) * 2.0}; // Cpp: {} same as Point2D because the compiler knows what to expect
}

std::vector<Point2D> generateData(int numInliers, int numOutliers, double noiseStddev,
                                   double xMin, double xMax, std::mt19937& rng) { // Cpp: std::mt19937 is a type, passed by reference, It produces the raw random bits you feed into distributions to get random numbers.
    std::vector<Point2D> points;
    points.reserve(numInliers + numOutliers);

    std::uniform_real_distribution<double> xDist(xMin, xMax);
    std::normal_distribution<double> noise(0.0, noiseStddev);
    for (int i = 0; i < numInliers; ++i) {
        const double x = xDist(rng);
        Point2D p = trueCurve(x);
        p.x += noise(rng);
        p.y += noise(rng);
        points.push_back(p);
    }

    std::uniform_real_distribution<double> yDist(-4.0, 4.0);
    for (int i = 0; i < numOutliers; ++i) {
        points.push_back({xDist(rng), yDist(rng)});
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

struct FitResult {
    std::vector<Point2D> controlPoints;
    std::vector<Point2D> curve;
    std::vector<bool> inlierMask;
    int inlierCount = 0;
};

FitResult fitRansac(const std::vector<Point2D>& data, int numberOfControlPoints, int tries, double tension,
                     double threshold, int samplesPerSegment, std::mt19937& rng) {
    FitResult best;

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

        std::vector<Point2D> curve = rsf::sampleCardinalSpline(controlPoints, tension, samplesPerSegment);

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
    const int numInliers = argc > 1 ? std::atoi(argv[1]) : 200; // Cpp: const: marks a variable as immutable after initialization — the compiler errors if you try to modify it later.
    const int numOutliers = argc > 2 ? std::atoi(argv[2]) : 80;
    const int numberOfControlPoints = argc > 3 ? std::atoi(argv[3]) : 6;
    const int tries = argc > 4 ? std::atoi(argv[4]) : 500;
    const double threshold = argc > 5 ? std::atof(argv[5]) : 0.3;
    const double tension = 0.5;
    const int samplesPerSegment = 20;

    std::mt19937 rng(42);

    const std::vector<Point2D> data = generateData(numInliers, numOutliers, 0.1, 0.0, 10.0, rng);
    const FitResult best = fitRansac(data, numberOfControlPoints, tries, tension, threshold, samplesPerSegment, rng);

    std::cout << "Data points: " << data.size() << " (" << numInliers << " generated inliers, "
              << numOutliers << " outliers)\n";
    std::cout << "Best fit: " << best.inlierCount << " / " << data.size() << " inliers using "
              << numberOfControlPoints << " control points over " << tries << " tries\numberOfControlPoints";

    writeCsv("data_points.csv", data, &best.inlierMask);
    writeCsv("spline_curve.csv", best.curve);
    writeCsv("control_points.csv", best.controlPoints);

    std::cout << "Wrote data_points.csv, spline_curve.csv, control_points.csv\n";
    return 0;
}
