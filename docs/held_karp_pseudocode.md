```
function orderPointsByProximity(_points):
    _n = number of points
    if _n < 3:
        return _points unchanged

    // 1. Precompute pairwise distances
    _dist[i][j] = distance(_points[i], _points[j])  for all i, j

    // 2. Dynamic programming over subsets (Held-Karp, open-path variant)
    // _bestLen[_S][_v] = length of the shortest path that visits exactly
    //                    the points in set _S, ending at point _v
    // _parent[_S][_v]  = the point visited right before _v on that best path

    for each point _i:
        _bestLen[{_i}][_i] = 0

    for each subset _S (smallest to largest):
        for each point _last in _S:
            for each point _next not in _S:
                _Sprime = _S + {_next}
                _candidate = _bestLen[_S][_last] + _dist[_last][_next]
                if _candidate < _bestLen[_Sprime][_next]:
                    _bestLen[_Sprime][_next] = _candidate
                    _parent[_Sprime][_next] = _last

    // 3. Find the best point to end the full path on
    _fullSet = set of all points
    _endPoint = the point _v that minimizes _bestLen[_fullSet][_v]

    // 4. Reconstruct the path by following parent pointers backward
    _path = []
    _S = _fullSet
    _v = _endPoint
    while _v is defined:
        prepend _v to _path
        _prev = _parent[_S][_v]
        remove _v from _S
        _v = _prev

    // 5. Normalize direction (two directions are equally short — pick one)
    if _path.last.x < _path.first.x, or (equal x and _path.last.y < _path.first.y):
        reverse(_path)

    return _path reordered as _points
```
