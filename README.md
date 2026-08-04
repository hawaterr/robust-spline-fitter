# robust-spline-fitter

The goal of this repo is to provide a python library written in C++ (for efficiency) that fit different spline types to data points. For now start with 2d data, maybe later extend to 3D or N dimensional. 

Plan:

- optimize more: why it takes 100 ms for 500 tries? compare with python code, caching, debug current change that makes performcance worse

- extend to 3/N dimensional data?
- another solver to min distance to spline: Ktrees, binary search, y difference(very fast), x difference, 
- more satisfies spacing constraints

- have it pip installable
- have python example scripts
- liscence: non commercial use only, may be used in research, no AI training, comes as is no garentee or liability...
- documentation: show off in readem with some graphs

- automatic finding of number of control points
- extend with other spline typese: faster linear splines, ..

- win by best inliers or best score, both options
- an online feature


Scope: start as a 2 day project, grow if I want to

