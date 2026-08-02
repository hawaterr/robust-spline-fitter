#!/usr/bin/env bash
set -e

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake -S "$repo_root" -B "$repo_root/build"
cmake --build "$repo_root/build"

cd "$repo_root/build"
./fit_spline
python3 "$repo_root/scripts/plot_result.py"
