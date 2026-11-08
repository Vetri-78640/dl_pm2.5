#!/bin/bash
# usage: bash wait_kernels.sh <poll_seconds> <max_polls> <owner/slug> [<owner/slug> ...]
# Returns 0 when every listed kernel has a final status (COMPLETE, ERROR or CANCEL*),
# printing each status. A failed status call counts as not finished. Exits 1 at the poll limit.
set -u
K="uvx --python 3.12 --from kaggle==2.2.5 --with six kaggle"
poll=$1; max=$2; shift 2
for i in $(seq 1 "$max"); do
  finished=1
  all=""
  for k in "$@"; do
    st=$($K kernels status "$k")
    all="$all $st"
    case "$st" in
      *COMPLETE*|*ERROR*|*CANCEL*) ;;
      *) finished=0 ;;
    esac
  done
  if [ "$finished" = 1 ]; then
    for k in "$@"; do echo "$(date +%H:%M) $($K kernels status "$k")"; done
    exit 0
  fi
  sleep "$poll"
done
echo "still running after $max polls:$all"
exit 1
