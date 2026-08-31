# Roadmap

PRIORITY I
- statisfies spacing: min x and y only in explicit, implicit only min d? asserts?
- newton solver seems to have a problem
- parallelize
- Local refinement pass after RANSAC: once you have inliers, do a least-squares polish of control points (currently it's pure random search — a final gradient/Nelder-Mead refit on the winning inlier set would likely improve fit quality for free).  win by best inliers or best score, both options, maybe not needed since there is a refinement pass after inlier pass.
- more satisfies-spacing constraints
- automatic finding of number of control points
- A warm_start/init_control_points param to seed RANSAC near a known-good guess instead of pure random sampling — useful for refitting similar data repeatedly (e.g. video frame sequences). online fitting, so we keep getting new points
- fewer points than control points error
- extend to 3/N dimensional data? -> new branch


WOULD BE NICE
- CI
- Confidence/uncertainty band on the fit (e.g., bootstrap resampling of inliers) — nice differentiator vs. plain scipy splines.
- Weighted fitting: accept per-point weights/uncertainties 
- Multi-curve fitting: sequential RANSAC (fit, remove inliers, refit on remainder) to handle data that's a mixture of multiple curves/clusters, not just one curve + noise.
- Closed/periodic curves: support fitting a loop (first and last control points connect) — relevant if users have orbit-like or cyclic data, and cardinal splines support this naturally.
- Verbose/diagnostic mode: return not just the winning candidate but score-vs-tries history, useful for tuning tries and threshold without guessing.
- benchmark against sklear, and different methods and data, run time ...



LOW PRIORITY
- optimize: cache polynomials and evaluate t, t**2, ... with coefficients pre-calculated? gain might be very small
- another solver to min distance to spline: k-d trees, binary search, x difference, y difference solved analytically not with samples




LONG TERM GOAL?
- extend with other spline types: faster linear splines, MINVO basis, bezier, ..
- Can turn this to a general spline fitting library, with all kinds of splines. Robust or not. 