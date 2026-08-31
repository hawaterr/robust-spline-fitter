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

# Extra args are forwarded to pytest, e.g. ./test.sh --plot-failures
#
# Two deliberate bits of isolation here:
#
#   .venv/bin/python3 - used if it exists, so you don't have to remember to
#   activate it. The system python3 has pytest 6.2.5, too old for the
#   installed anyio (its plugin imports _pytest.scope, added in pytest 7).
#   Only the test run switches interpreters; the build above stays on
#   whatever python3 CMake configured against, so the .so keeps matching.
#
#   PYTHONPATH set to exactly $repo_root, NOT appended to the inherited one.
#   Sourcing /opt/ros/humble puts its site-packages on PYTHONPATH, which
#   auto-loads eight ROS pytest plugins (launch_testing, ament_*, ...) plus
#   anyio into every run; launch_testing then dies importing yaml. PYTHONPATH
#   also defeats venv isolation, so dropping it is what makes the venv stick.
test_python="$repo_root/.venv/bin/python3"
[ -x "$test_python" ] || test_python="python3"

PYTHONPATH="$repo_root" "$test_python" -m pytest "$repo_root/tests" "$@"
