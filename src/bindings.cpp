#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rsf/ransac_fitter.hpp"

namespace py = pybind11;

namespace {

// Python-facing result type: plain lists/tuples instead of rsf::Point2D, so
// callers don't need to know about the C++ point type.
struct PyFitResult {
    std::vector<std::pair<double, double>> control_points;
    std::vector<std::pair<double, double>> curve;
    std::vector<bool> inlier_mask;
    int inlier_count = 0;
};

std::vector<std::pair<double, double>> toPairs(const std::vector<rsf::Point2D>& points) {
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(points.size());
    for (const auto& p : points) pairs.emplace_back(p.x, p.y);
    return pairs;
}

PyFitResult fit_ransac(const std::vector<double>& x, const std::vector<double>& y,
                        const rsf::RansacFitParams& params, unsigned int seed) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same length");
    }

    std::vector<rsf::Point2D> data;
    data.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        data.push_back({x[i], y[i]});
    }

    std::mt19937 rng(seed);
    const rsf::FitResult result = rsf::fitRansac(data, params, rng);

    PyFitResult pyResult;
    pyResult.control_points = toPairs(result.controlPoints);
    pyResult.curve = toPairs(result.curve);
    pyResult.inlier_mask = result.inlierMask;
    pyResult.inlier_count = result.inlierCount;
    return pyResult;
}

}  // namespace

PYBIND11_MODULE(rsf, m) { // so we say import rsf in python
    m.doc() = "Robust cardinal spline fitting via RANSAC";

    py::enum_<rsf::DistanceMetric>(m, "DistanceMetric")
        .value("Perpendicular", rsf::DistanceMetric::Perpendicular)
        .value("Vertical", rsf::DistanceMetric::Vertical);

    py::class_<rsf::RansacFitParams>(m, "RansacFitParams") // binding input
        .def(py::init<>()) // can do in python params = rsf.RansacFitParams()
        .def_readwrite("numberOfControlPoints", &rsf::RansacFitParams::numberOfControlPoints) // can do params.numberOfControlPoints = self.control_points
        .def_readwrite("tries", &rsf::RansacFitParams::tries)
        .def_readwrite("tension", &rsf::RansacFitParams::tension)
        .def_readwrite("threshold", &rsf::RansacFitParams::threshold)
        .def_readwrite("samplesPerSegment", &rsf::RansacFitParams::samplesPerSegment)
        .def_readwrite("minControlPointXGap", &rsf::RansacFitParams::minControlPointXGap)
        .def_readwrite("maxControlPointYGap", &rsf::RansacFitParams::maxControlPointYGap)
        .def_readwrite("distanceMetric", &rsf::RansacFitParams::distanceMetric);

    py::class_<PyFitResult>(m, "FitResult") // binding output
        .def_readonly("control_points", &PyFitResult::control_points) // can do control_points_ = result.control_points
        .def_readonly("curve", &PyFitResult::curve)
        .def_readonly("inlier_mask", &PyFitResult::inlier_mask)
        .def_readonly("inlier_count", &PyFitResult::inlier_count);

    m.def("fit_ransac", &fit_ransac, py::arg("x"), py::arg("y"), py::arg("params") = rsf::RansacFitParams(),
          py::arg("seed") = 42,
          "Robustly fit a cardinal spline to (x, y) data via RANSAC, returning the best FitResult.");
}
