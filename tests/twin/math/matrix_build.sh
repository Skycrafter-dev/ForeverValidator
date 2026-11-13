#!/usr/bin/env bash
# Configure and build every variant of the twin math tests.
#   tests/twin/math/matrix_build.sh [outdir]      (default ~/fv-twin-math-build)
# Variants: {gcc, clang, mingw} x {sse2, avx2, avx512}, gcc/clang -march=native,
# gcc -O0, clang -O3, and two negative controls built with contraction
# (-ffp-contract=fast -mfma) that must FAIL the canary and change the hashes.
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$HOME/fv-twin-math-build}"
jobs="${TW_JOBS:-4}"
mkdir -p "$out"

conf() {  # name, extra cmake args...
  local name="$1"; shift
  nice -n 10 cmake -S "$src" -B "$out/$name" -G Ninja "$@" > "$out/$name.configure.log" 2>&1 \
    || { echo "configure $name failed (see $out/$name.configure.log)"; return 1; }
  nice -n 10 ninja -C "$out/$name" -j "$jobs" > "$out/$name.build.log" 2>&1 \
    || { echo "build $name failed (see $out/$name.build.log)"; return 1; }
  echo "built $name"
}

for isa in sse2 avx2 avx512; do
  conf "gcc-$isa"   -DCMAKE_CXX_COMPILER=g++     -DTW_MATH_ISA=$isa
  conf "clang-$isa" -DCMAKE_CXX_COMPILER=clang++ -DTW_MATH_ISA=$isa
  if command -v x86_64-w64-mingw32-g++-posix > /dev/null; then
    conf "mingw-$isa" -DCMAKE_TOOLCHAIN_FILE="$src/mingw-w64.cmake" -DTW_MATH_ISA=$isa
  fi
done
conf gcc-native   -DCMAKE_CXX_COMPILER=g++     -DTW_MATH_ISA=native
conf clang-native -DCMAKE_CXX_COMPILER=clang++ -DTW_MATH_ISA=native
conf gcc-avx2-O0  -DCMAKE_CXX_COMPILER=g++     -DTW_MATH_ISA=avx2 -DTW_MATH_OPT=-O0
conf clang-avx512-O3 -DCMAKE_CXX_COMPILER=clang++ -DTW_MATH_ISA=avx512 -DTW_MATH_OPT=-O3
conf gcc-avx2-fma-contract   -DCMAKE_CXX_COMPILER=g++     -DTW_MATH_ISA=avx2 -DTW_MATH_CONTRACT=fast
conf clang-avx2-fma-contract -DCMAKE_CXX_COMPILER=clang++ -DTW_MATH_ISA=avx2 -DTW_MATH_CONTRACT=fast
