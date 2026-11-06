#!/bin/bash
# Usage: run_sandbox.sh DEST SCRIPT...   Runs each script from its sandbox copy, with the sandbox as the working directory.
d="$1"; shift; cd "$d" || exit 1
for s in "$@"; do
  t0=$(date +%s); ~/ip_llm/bin/python "$s.py" > "log_$s.txt" 2>&1; rc=$?
  echo "$s rc=$rc $(( $(date +%s) - t0 ))s"
done
