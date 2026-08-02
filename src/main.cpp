#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rsf/ransac_fitter.hpp"

using rsf::Point2D;

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

int main() {
    const std::string inputPath = "../data/data.csv";

    rsf::RansacFitParams params;
    params.numberOfControlPoints = 4;
    params.tries = 5000;
    params.threshold = 0.2;
    params.tension = 0.5;
    params.samplesPerSegment = 200;
    params.minControlPointXGap = 1.0;
    params.maxControlPointYGap = 2.0;

    std::mt19937 rng(42);

    const std::vector<Point2D> data = loadCsv(inputPath);
    const rsf::FitResult best = rsf::fitRansac(data, params, rng);

    std::cout << "Loaded " << data.size() << " points from " << inputPath << "\n";

    if (best.controlPoints.empty()) {
        std::cout << "No candidate spline found: every sampled control-point set was rejected "
                     "(check minControlPointXGap/maxControlPointYGap against the data's scale).\n";
        return 1;
    }

    std::cout << "Best fit: " << best.inlierCount << " / " << data.size() << " inliers using "
              << params.numberOfControlPoints << " control points over " << params.tries << " tries\n";

    writeCsv("data_points.csv", data, &best.inlierMask);
    writeCsv("spline_curve.csv", best.curve);
    writeCsv("control_points.csv", best.controlPoints);

    std::cout << "Wrote data_points.csv, spline_curve.csv, control_points.csv\n";
    return 0;
}
