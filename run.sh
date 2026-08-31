#!/usr/bin/env bash
set -e

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "$repo_root" -B "$repo_root/build"
cmake --build "$repo_root/build"

# regressor.py does `from . import rsf`, which requires the compiled extension
# to sit inside the package directory. CMake builds it to build/ for local
# dev; symlink it into place so the same relative import works both here and
# once actually pip-installed (where it's a real installed sibling file).
rsf_so="$(find "$repo_root/build" -maxdepth 1 -name 'rsf.cpython-*.so' | head -n1)"
ln -sf "$rsf_so" "$repo_root/robust_spline_fitter/$(basename "$rsf_so")"

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 "$repo_root/examples/example.py"
