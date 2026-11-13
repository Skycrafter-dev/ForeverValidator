#!/usr/bin/env bash
# Object-code audit of every twin library object in a build (design.md §7):
#   - FAIL on any FMA (vfmadd/vfmsub/vfnmadd/vfnmsub) or approximate
#     reciprocal (rcpps/rsqrtps/vrcp14/vrsqrt14/vrcp28/vrsqrt28) instruction
#     in any twin object (the twin's arithmetic is +-*/sqrt only, math.md §4);
#   - FAIL on an undefined inexact libm symbol (transcendentals, pow, fma)
#     in tick-path objects (tw_engine, tw_collide, tw_car, tw_isa and its ISA
#     variants, the math audit instance). Exact IEEE functions (sqrt, fmin,
#     fmax, floor, ...) are listed but allowed: their result is defined bit
#     for bit. Host-side objects (pack and world compile, control-stream
#     builder) are listed for information only.
#   audit_twin_objects.sh <build dir> [--strict]
# Gate (user rule of 2026-11-09, no FMA): exit 1 on any FMA / rcp / rsqrt instruction in any
# twin object. Inexact libm calls in the tick path are listed for information (the same libm
# on the same machine keeps same-binary repeatability); --strict also fails on them.
# Uses the same patterns as tests/twin/math/audit_objects.sh.
set -uo pipefail
dir="$1"; strict=0; [ "${2:-}" = "--strict" ] && strict=1
objdump=objdump; nm=nm
if command -v x86_64-w64-mingw32-objdump > /dev/null && [[ "$(basename "$dir")" == mingw-* ]]; then
  objdump=x86_64-w64-mingw32-objdump; nm=x86_64-w64-mingw32-nm
fi
ins_re='\s(v?fn?m(add|sub)[0-9a-z]*|v?rcp(ps|ss|14|28)[a-z]*|v?rsqrt(ps|ss|14|28)[a-z]*)\s'
libm_re='\b_?_?(sin|cos|tan|asin|acos|atan|atan2|exp|exp2|expm1|log|log2|log10|log1p|pow|sqrt|fmod|floor|ceil|trunc|round|lround|rint|lrint|nearbyint|hypot|fma|sincos|ldexp|frexp|modf|copysign|fabs|fmin|fmax)(f|l)?\b'
inexact_re='\b_?_?(sin|cos|tan|asin|acos|atan|atan2|exp|exp2|expm1|log|log2|log10|log1p|pow|hypot|fma|sincos)(f|l)?\b'
bad=0; found=0
for lib in tw_engine tw_collide tw_car tw_isa tw_math_audit tw_race tw_world tw_model; do
  while IFS= read -r obj; do
    [ -n "$obj" ] || continue
    found=$((found + 1))
    n_ins=$($objdump -d --no-show-raw-insn "$obj" | grep -E -c "$ins_re")
    n_libm=$($nm -u "$obj" | grep -E -c "$libm_re")
    total=$($objdump -d --no-show-raw-insn "$obj" | grep -c -E '^\s+[0-9a-f]+:')
    tick=no; case "$lib" in tw_engine|tw_collide|tw_car|tw_isa|tw_math_audit) tick=yes;; esac
    verdict=ok
    if [ "$n_ins" != 0 ]; then verdict=FAIL; bad=1; fi
    if [ "$n_libm" != 0 ]; then
      n_inexact=$($nm -u "$obj" | grep -E -c "$inexact_re")
      names=$($nm -u "$obj" | grep -E -o "$libm_re" | sort -u | tr '\n' ' ')
      if [ "$tick" = yes ] && [ "$n_inexact" != 0 ]; then
        if [ "$strict" = 1 ]; then verdict=FAIL; bad=1; else verdict="$verdict (inexact libm, info: $names)"; fi
      else verdict="$verdict ($([ "$tick" = yes ] && echo "exact libm" || echo "host libm"): $names)"; fi
    fi
    printf '%-14s %-28s %7s insns  fma/rcp %-3s  libm %-3s  %s\n' "$lib" "$(basename "$obj")" "$total" "$n_ins" "$n_libm" "$verdict"
    [ "$n_ins" = 0 ] || $objdump -d --no-show-raw-insn "$obj" | grep -E "$ins_re" | head -3
  done < <(find "$dir" -path "*CMakeFiles/$lib.dir/*" \( -name '*.o' -o -name '*.obj' \) 2>/dev/null | sort)
done
# The isolated ISA variant objects (isa/isolate_variant.sh): tick path.
while IFS= read -r obj; do
  [ -n "$obj" ] || continue
  found=$((found + 1))
  n_ins=$($objdump -d --no-show-raw-insn "$obj" | grep -E -c "$ins_re")
  n_libm=$($nm -u "$obj" | grep -E -c "$libm_re")
  total=$($objdump -d --no-show-raw-insn "$obj" | grep -c -E '^\s+[0-9a-f]+:')
  verdict=ok
  n_inexact=$($nm -u "$obj" | grep -E -c "$inexact_re")
  names=$($nm -u "$obj" | grep -E -o "$libm_re" | sort -u | tr '\n' ' ')
  if [ "$n_ins" != 0 ] || { [ "$strict" = 1 ] && [ "$n_inexact" != 0 ]; }; then verdict=FAIL; bad=1;
  elif [ "$n_inexact" != 0 ]; then verdict="ok (inexact libm, info: $names)";
  elif [ "$n_libm" != 0 ]; then verdict="ok (exact libm: $names)"; fi
  printf '%-14s %-28s %7s insns  fma/rcp %-3s  libm %-3s  %s\n' "isa-variant" "$(basename "$obj")" "$total" "$n_ins" "$n_libm" "$verdict"
done < <(find "$dir" -maxdepth 1 -name 'tw_isa_*.o' 2>/dev/null | sort)
[ "$found" -gt 0 ] || { echo "no twin objects under $dir"; exit 2; }
[ "$bad" = 0 ] && echo "audit: PASS ($found objects)" || echo "audit: FAIL"
exit $bad
