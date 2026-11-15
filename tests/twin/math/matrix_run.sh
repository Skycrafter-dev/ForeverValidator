#!/usr/bin/env bash
# Run the twin math tests over built variants (see matrix_build.sh) and compare.
#   matrix_run.sh identity [builds...]              canary + identity hash per build
#   matrix_run.sh exhaustive "<fn> <fn>" [builds...] all 2^32 inputs per function (3-10 s each)
# Builds default to every directory in $TW_OUT (~/fv-twin-math-build). mingw-*
# builds run under wine with an isolated prefix ($TW_WINEPREFIX, default
# ~/fv-twin-math-wine). Keep each invocation under 60 s: pass a subset (about
# 4-6 s per optimised build incl. mingw under wine, 25 s for -O0).
set -uo pipefail
out="${TW_OUT:-$HOME/fv-twin-math-build}"
export WINEPREFIX="${TW_WINEPREFIX:-$HOME/fv-twin-math-wine}" WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="
mode="${1:-identity}"; shift || true
fns=""
if [ "$mode" = exhaustive ]; then fns="$1"; shift; fi
builds=("$@")
if [ ${#builds[@]} -eq 0 ]; then
  for d in "$out"/*/; do [ -f "$d/CMakeCache.txt" ] && builds+=("$(basename "$d")"); done
fi

run() {  # build, args...
  local b="$1"; shift
  if [[ "$b" == *-O0 ]]; then export TW_NATIVE_LANES_ONLY=1; else unset TW_NATIVE_LANES_ONLY; fi
  if [[ "$b" == mingw-* ]]; then
    (cd "$out/$b" && timeout 55 nice -n 10 wine64 ./tw_math_tests.exe "$@" 2>&1 | tr -d '\r')
  else
    (cd "$out/$b" && timeout 55 nice -n 10 ./tw_math_tests "$@" 2>&1)
  fi
}

if [ "$mode" = identity ]; then
  printf '%-26s %-8s %-18s %-6s %-8s\n' build canary identity lanes ftz-guard
  for b in "${builds[@]}"; do
    c=$(run "$b" canary | grep -c DETECTED)
    id=$(run "$b" identity)
    h=$(echo "$id" | awk '/identity TOTAL/{print $3}')
    l=$(echo "$id" | awk '/identity TOTAL/{print $NF}')
    g=$(echo "$id" | awk '/FpGuard/{print $NF}')
    printf '%-26s %-8s %-18s %-6s %-8s\n' "$b" "$([ "$c" = 0 ] && echo ok || echo FUSED)" "$h" "$l" "${g:-n/a}"
    echo "$id" | grep '^identity ' | grep -v TOTAL | awk -v b="$b" '{print b, $2, $3}' >> "$out/identity-per-function.txt"
  done
else
  for b in "${builds[@]}"; do
    for f in $fns; do
      r=$(run "$b" exhaustive "$f")
      echo "$b $r"
    done
  done
fi
