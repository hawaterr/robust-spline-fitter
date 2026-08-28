# Finding the closest point on a spline, without sampling it

For a query point `Q`, the closest point on the curve satifies:

```
f(t) = (P(t) − Q) · P′(t) = 0
```



## Solving it: Newton's method

`f(t)` doesn't have a convenient closed-form root, but Newton's method
converges to one in a handful of steps given a decent starting guess:

```
t_next = t − f(t) / f′(t)
```
ADD EXPANDED

each step clamped back into `[0, 1]`, since a segment only exists on that
range. This is `singleNewtonRun` in `spline_distance_analytic.cpp`.

To avoid getting stuck at a local minima, run Newton from six different starting points per segment (t = 0, 0.25, 0.5, 0.75, 1, plus a guess proportional to where the query's x falls)

## Assembling the full answer

A fitted spline is several segments plus two straight-line extensions at the
ends (so the curve spans the data's full x-range). We check all of them and get the minimum. 
ADD LINEAR EXTENSIONS
