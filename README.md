# robust-spline-fitter

The goal of this repo is to provide a python library written in C++ (for efficiency) that fit different spline types to data points. For now start with 2d data, maybe later extend to 3D or N dimensional. 

Plan:


- have python example scripts
- refactor to better file and folder structure
- liscence: non commercial use only, may be used in research, no AI training, comes as is no garentee or liability...
- documentation: show off in readem with some graphs, more data ...
- have it pip installable


- more satisfies spacing constraints
- extend to 3/N dimensional data?

- automatic finding of number of control points
- extend with other spline types: faster linear splines, ..

- optimize: cache polynomials and evaluate t t**2 ,... with coeaff pre calculated?
- another solver to min distance to spline: Ktrees, binary search, x difference, 

- win by best inliers or best score, both options
- an online feature

