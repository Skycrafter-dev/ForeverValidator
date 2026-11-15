#!/usr/bin/env bash
# Twin determinism gate: the same binary on the same CPU repeats bit for bit.
#
#   repeat.sh COMMAND [ARG...]
#
# Runs COMMAND twice (two processes) and compares the two outputs byte for
# byte. COMMAND prints its results as hashes or records (tw_golden_hash run,
# twin_harness run --out -, ...). Exit 0 when both runs succeed and agree.
set -uo pipefail
a=$("$@" 2>/dev/null); ra=$?
b=$("$@" 2>/dev/null); rb=$?
if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then
  echo "repeat: exit codes $ra / $rb: $*"; exit 1
fi
if [ "$a" != "$b" ]; then
  echo "repeat: DIFFERENT outputs: $*"
  diff <(echo "$a") <(echo "$b") | head -10
  exit 1
fi
echo "repeat: same ($(echo "$a" | wc -l) lines): $*"
