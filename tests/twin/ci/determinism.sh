#!/usr/bin/env bash
# Twin determinism CI (design.md §7, milestone M9).
#
# Rule (user decisions 2026-11-08 / 2026-11-09): no FMA (-ffp-contract=off on every twin TU);
# -march=x86-64-v3/-v4 and LTO per ISA variant are allowed (src/twin/CMakeLists.txt defaults).
# Gates, in every build:
#   repeat  the same binary on the same CPU repeats bit for bit: tw_golden_hash run components
#           and run isa twice, outputs compared (tests/twin/ci/repeat.sh);
#   audit   no FMA / rcp / rsqrt instruction in any twin object (audit_twin_objects.sh; inexact
#           libm calls are listed, and fail only in -strict builds);
#   hashes  the component hashes equal tests/twin/golden/goldens.txt and the first build, and
#           every ISA variant's selftest equals the baseline's (no contraction: the same bits).
# Also printed: the math canary + identity hash (tests/twin/math/matrix_run.sh).
#
#   determinism.sh build [BUILD...]   configure + compile (compile time is not bounded)
#   determinism.sh run   [BUILD...]   the checks, a few seconds per build; keep one call under 60 s
#   determinism.sh list               the build names
#
# Builds: gcc-{sse2,avx2,avx512}, clang-{sse2,avx2,avx512}, gcc/clang-default
# (dispatch: all three tick variants in one binary) by default; gcc-default-strict and
# clang-default-strict (no -march, no LTO, libm audited) by name.
# Environment: TW_OUT (default ~/fv-twin-int-build/ci), TW_JOBS (default 3).
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${TW_OUT:-$HOME/fv-twin-int-build/ci}"
jobs="${TW_JOBS:-3}"
# *-default: baseline ISA with runtime dispatch, so the AVX2 and AVX-512
# variants of the tick path are built into the same binary and checked
# against the baseline (stage isa).
default_builds=(gcc-sse2 gcc-avx2 gcc-avx512 clang-sse2 clang-avx2 clang-avx512 gcc-default clang-default)
run_cpu=(nice -n 10 taskset -c 0-4)

mode="${1:-}"; shift || true
builds=("$@")
[ ${#builds[@]} -gt 0 ] || builds=("${default_builds[@]}")

config_args() {  # build name -> cmake arguments
  local name="$1" cc isa
  local extra="-DFOREVERVALIDATOR_TWIN_CONTRACT=off -DFOREVERVALIDATOR_TWIN_MARCH=ON -DFOREVERVALIDATOR_TWIN_LTO=ON"
  cc="${name%%-*}"; isa="${name#*-}"; isa="${isa%%-*}"
  [[ "$name" == *-strict ]] && extra="-DFOREVERVALIDATOR_TWIN_CONTRACT=off -DFOREVERVALIDATOR_TWIN_MARCH=OFF -DFOREVERVALIDATOR_TWIN_LTO=OFF"
  case "$cc" in gcc) echo -n "-DCMAKE_CXX_COMPILER=g++ ";; clang) echo -n "-DCMAKE_CXX_COMPILER=clang++ ";;
    *) echo "unknown compiler in $name" >&2; return 1;; esac
  echo "-DFOREVERVALIDATOR_TWIN_ISA=$isa $extra"
}

case "$mode" in
list)
  printf '%s\n' "${default_builds[@]}" gcc-default-strict clang-default-strict
  ;;
build)
  mkdir -p "$out"
  status=0
  for b in "${builds[@]}"; do
    rm -f "$out/$b.ok"
    args=$(config_args "$b") || exit 2
    # shellcheck disable=SC2086
    "${run_cpu[@]}" cmake -S "$repo/src/twin" -B "$out/$b" -G Ninja $args > "$out/$b.configure.log" 2>&1 \
      || { echo "configure $b failed ($out/$b.configure.log)"; exit 1; }
    if "${run_cpu[@]}" ninja -C "$out/$b" -j "$jobs" > "$out/$b.build.log" 2>&1; then
      touch "$out/$b.ok"; echo "built $b"
    else
      echo "build $b failed ($out/$b.build.log)"; status=1
    fi
  done
  exit $status
  ;;
run)
  rm -f "$out/identity-per-function.txt"
  echo "== math canary + identity (informational; tests/twin/math/matrix_run.sh)"
  TW_OUT="$out" "${run_cpu[@]}" "$here/../math/matrix_run.sh" identity "${builds[@]}" || true
  echo "== repeat gate + hashes"
  ref=""; fail=0
  printf '%-22s %-8s %-10s %-10s %-9s %s\n' build repeat components isa audit vs-first
  for b in "${builds[@]}"; do
    d="$out/$b"
    # Only builds whose last `build` succeeded: a stale binary proves nothing.
    [ -x "$d/tw_golden_hash" ] && [ -f "$out/$b.ok" ] || { echo "$b: not built (or its last build failed)"; fail=1; continue; }
    strict=0; [[ "$b" == *-strict ]] && strict=1
    (cd "$d" && "${run_cpu[@]}" timeout 25 "$here/repeat.sh" ./tw_golden_hash run components > repeat.txt 2>&1 &&
                "${run_cpu[@]}" timeout 25 "$here/repeat.sh" ./tw_golden_hash run isa >> repeat.txt 2>&1)
    r=$?
    (cd "$d" && "${run_cpu[@]}" timeout 25 ./tw_golden_hash run components > golden.components.txt 2> golden.err)
    (cd "$d" && "${run_cpu[@]}" timeout 25 ./tw_golden_hash check components > golden.check.txt 2>> golden.err)
    cc=$?
    (cd "$d" && "${run_cpu[@]}" timeout 25 ./tw_golden_hash check isa > golden.isa.txt 2>> golden.err)
    i=$?
    if [ $strict = 1 ]; then "${run_cpu[@]}" "$here/audit_twin_objects.sh" "$d" --strict > "$d/audit.txt" 2>&1
    else "${run_cpu[@]}" "$here/audit_twin_objects.sh" "$d" > "$d/audit.txt" 2>&1; fi
    a=$?
    { cat "$d/golden.components.txt"; grep -E '^(ok|DIFF|MISSING) ' "$d/golden.isa.txt"; } > "$d/golden.all.txt"
    if [ -z "$ref" ]; then ref="$d/golden.all.txt"; same="(reference)";
    elif cmp -s "$ref" "$d/golden.all.txt"; then same=same; else same=DIFFERENT; fail=1; fi
    rep=$([ $r = 0 ] && echo ok || echo FAIL)
    comp=$([ $cc = 0 ] && echo pinned || echo DIFFERENT)
    isa=$([ $i = 0 ] && echo equal || echo DIFFERENT)
    aud=$([ $a = 0 ] && echo clean || echo FAIL)
    [ "$rep" = ok ] && [ $cc = 0 ] && [ $i = 0 ] && [ $a = 0 ] || fail=1
    printf '%-22s %-8s %-10s %-10s %-9s %s\n' "$b" "$rep" "$comp" "$isa" "$aud" "$same"
    [ "$rep" = ok ] || sed 's/^/   /' "$d/repeat.txt"
  done
  echo "-- repeat: same binary twice on this CPU; components: against tests/twin/golden/goldens.txt;"
  echo "   isa: every variant against the baseline; audit: no FMA / rcp (all gates)"
  echo "-- first build:"; sed 's/^/   /' "$ref" 2>/dev/null
  [ $fail = 0 ] && echo "determinism: PASS (${#builds[@]} builds)" || echo "determinism: FAIL"
  exit $fail
  ;;
*)
  sed -n '2,22p' "$0"; exit 2
  ;;
esac
