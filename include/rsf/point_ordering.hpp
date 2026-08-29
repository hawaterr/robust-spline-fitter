#pragma once

#include <vector>

#include "rsf/point.hpp"

namespace rsf {

// Reorders points so the total length of the path through them is minimal:
// an open TSP (shortest Hamiltonian path, no return to the start), solved
// exactly via Held-Karp in O(2^n * n^2).
//
// Used to order spline control points along the curve without assuming the
// curve is a function of x, so candidates that fold back in x (near-vertical
// runs, C shapes) can be represented at all.
//
// Intended for the small n a spline uses (4-8 control points). The cost grows
// sharply beyond that: measured per call, ~0.4us at n=4, ~46us at n=8,
// ~284us at n=10.
//
// The resulting path is normalized to run in the direction of increasing x
// (tie-broken on y), since an open path's two directions have equal length.
void orderPointsByProximity(std::vector<Point2D>& points);

}  // namespace rsf
