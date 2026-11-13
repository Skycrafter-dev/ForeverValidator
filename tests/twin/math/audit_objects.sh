#!/usr/bin/env bash
# Audit the compiled math object of one build:
#   - no FMA (vfmadd/vfmsub/vfnmadd/vfnmsub), no approximate reciprocal
#     (rcpps/rsqrtps/vrcp14/vrsqrt14/vrcp28/vrsqrt28) instructions
#   - no undefined math-library symbol (sinf, cosf, expf, powf, ...)
#   audit_objects.sh <build dir>     (exit 1 on any finding)
set -uo pipefail
dir="$1"
obj=$(find "$dir" -name 'tw_math_audit.cpp.o*' -o -name 'tw_math_audit.cpp.obj' | head -1)
[ -n "$obj" ] || { echo "no audit object in $dir"; exit 2; }
objdump=objdump; nm=nm
if [[ "$(basename "$dir")" == mingw-* ]]; then objdump=x86_64-w64-mingw32-objdump; nm=x86_64-w64-mingw32-nm; fi
bad=0
ins=$($objdump -d --no-show-raw-insn "$obj" | grep -E -c '\s(v?fn?m(add|sub)[0-9a-z]*|v?rcp(ps|ss|14|28)[a-z]*|v?rsqrt(ps|ss|14|28)[a-z]*)\s')
fns=$($nm -u "$obj" | grep -E -c '\b_?_?(sin|cos|tan|asin|acos|atan|atan2|exp|exp2|expm1|log|log2|log10|log1p|pow|sqrt|fmod|floor|ceil|trunc|round|lround|rint|lrint|nearbyint|hypot|fma|sincos|ldexp|frexp|modf|copysign|fabs|fmin|fmax)(f|l)?\b')
total=$($objdump -d --no-show-raw-insn "$obj" | grep -c -E '^\s+[0-9a-f]+:')
echo "$(basename "$dir"): $total instructions, fma/rcp/rsqrt: $ins, libm symbols: $fns"
[ "$ins" = 0 ] || { $objdump -d --no-show-raw-insn "$obj" | grep -E '\s(v?fn?m(add|sub)|v?rcp|v?rsqrt)' | head -5; bad=1; }
[ "$fns" = 0 ] || { $nm -u "$obj"; bad=1; }
exit $bad
