#!/usr/bin/env bash
# Twin ISA dispatch: makes one self-contained variant object (tw_isa.h).
#
#   isolate_variant.sh OUT ENTRY SUFFIX OBJCOPY NM READELF -- LINK... -- OBJECT...
#
# 1. Partial-links the variant's objects into one relocatable object with
#    LINK... -o OUT.partial.o OBJECT... (`ld -r`, or the compiler driver's
#    partial LTO link: `g++ <flags> -flto -r -nostdlib -flinker-output=nolto-rel`).
# 2. Renames every defined global symbol and every COMDAT group signature to
#    NAME+SUFFIX, except ENTRY. Inline functions and template instances
#    compiled for this ISA (std:: included) therefore never merge with the
#    baseline's or another variant's copies at the final link. Localizing
#    them instead is not enough: GNU ld then still folds the groups by
#    signature and the program crashes.
# 3. Checks that ENTRY is the variant's only original global name.
set -euo pipefail
out="$1"; entry="$2"; suffix="$3"; objcopy="$4"; nm="$5"; readelf="$6"; shift 6
[ "${1:-}" = "--" ] || { echo "isolate_variant: expected -- before the link command" >&2; exit 2; }
shift
link=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do link+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "isolate_variant: expected -- before the objects" >&2; exit 2; }
shift
tmp="$out.partial.o"
"${link[@]}" -o "$tmp" "$@"
{ "$nm" --defined-only -g "$tmp" | awk '{print $NF}'
  "$readelf" -gW "$tmp" | sed -n 's/.*\[\(.*\)\] contains.*/\1/p'; } \
  | grep -v -x -F "$entry" | sort -u | awk -v s="$suffix" '{print $1 " " $1 s}' > "$out.redefine"
"$objcopy" --redefine-syms="$out.redefine" "$tmp" "$out"
rm -f "$tmp"
if ! "$nm" --defined-only -g "$out" | awk '{print $NF}' | grep -q -x -F "$entry"; then
  echo "isolate_variant: $entry is not defined in $out" >&2; exit 1
fi
leaked=$("$nm" --defined-only -g "$out" | awk '{print $NF}' | grep -v -x -F "$entry" | grep -v -F -- "$suffix" | head -3 || true)
if [ -n "$leaked" ]; then
  echo "isolate_variant: symbols left without the suffix in $out: $leaked" >&2; exit 1
fi
