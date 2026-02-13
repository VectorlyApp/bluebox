#!/bin/bash
# Parallel test: 25 Mass Corp Search routines with random 0-25s jitter via sidecar
#
# Prerequisites:
#   1. Chrome running on port 9222
#   2. Sidecar running: python cdp_proxy_sidecar.py --listen-port 9223 --chrome-port 9222

PROXIES=(
  "68.182.104.202:56870:buykvgtw:t669WYCuRalP"
  "68.182.104.122:49653:buykvgtw:t669WYCuRalP"
  "68.182.110.85:58409:buykvgtw:t669WYCuRalP"
  "68.182.109.0:60473:buykvgtw:t669WYCuRalP"
  "68.182.108.66:54179:buykvgtw:t669WYCuRalP"
)

TOTAL=25
MAX_JITTER=100

SIDECAR_PORT=9223
SIDECAR_ADDR="http://127.0.0.1:${SIDECAR_PORT}"
ROUTINE="example_data/example_routines/massachusetts_corp_search_routine.json"
PARAMS='{"entity_name": "Microsoft"}'
TMPDIR=$(mktemp -d)

# Check sidecar is running
if ! curl -s "http://127.0.0.1:${SIDECAR_PORT}/json/version" > /dev/null 2>&1; then
  echo "ERROR: Sidecar not running on port ${SIDECAR_PORT}"
  echo "Start it with: python cdp_proxy_sidecar.py --listen-port ${SIDECAR_PORT} --chrome-port 9222"
  exit 1
fi

echo "Mass Corp Search (Microsoft) — ${TOTAL} ROUTINES, random 0-${MAX_JITTER}s jitter"
echo "========================================================================"

total_start=$(python3 -c 'import time; print(time.time())')

# Launch all 25 at once, each with random jitter
for idx in $(seq 0 $((TOTAL - 1))); do
  proxy_idx=$((idx % ${#PROXIES[@]}))
  entry="${PROXIES[$proxy_idx]}"
  IFS=':' read -r host port user pass <<< "$entry"
  addr="${user}:${pass}@${host}:${port}"

  (
    jitter=$(python3 -c "import random; print(round(random.uniform(0, $MAX_JITTER), 2))")
    echo "JITTER ${jitter}" > "${TMPDIR}/meta_${idx}.txt"

    python3 -c "import time; time.sleep($jitter)"

    start=$(python3 -c 'import time; print(time.time())')

    output=$(bluebox-execute \
      --routine-path "$ROUTINE" \
      --parameters-dict "$PARAMS" \
      --proxy-address "$addr" \
      --remote-debugging-address "$SIDECAR_ADDR" \
      2>&1)

    end=$(python3 -c 'import time; print(time.time())')
    elapsed=$(python3 -c "print(round($end - $start, 2))")

    ok=$(echo "$output" | grep "ok=True" | tail -1)
    status="FAIL"
    [ -n "$ok" ] && status="OK"

    echo "${status} ${elapsed} ${jitter} ${host}:${port}" > "${TMPDIR}/result_${idx}.txt"
  ) &
done

echo "All ${TOTAL} launched (staggered by jitter). Waiting..."
echo ""
wait

total_end=$(python3 -c 'import time; print(time.time())')
wall_time=$(python3 -c "print(round($total_end - $total_start, 2))")

# Collect results
printf "%-5s %-3s %7s %8s  %s\n" "Run" "OK" "Time" "Jitter" "Proxy"
echo "------------------------------------------------------------------------"

sum=0
passed=0
failed=0
for idx in $(seq 0 $((TOTAL - 1))); do
  read -r status elapsed jitter proxy < "${TMPDIR}/result_${idx}.txt"
  if [ "$status" = "OK" ]; then
    icon="✅"
    ((passed++))
  else
    icon="❌"
    ((failed++))
  fi
  printf "%-5s %s  %6ss  %5ss  %s\n" "$((idx+1))" "$icon" "$elapsed" "$jitter" "$proxy"
  sum=$(python3 -c "print(round($sum + $elapsed, 2))")
done

avg=$(python3 -c "print(round($sum / $TOTAL, 2))")

echo "------------------------------------------------------------------------"
echo "Passed: $passed  Failed: $failed  Total: $TOTAL"
echo "Sum of individual times: ${sum}s  Avg per routine: ${avg}s"
echo ""
echo ">>> Wall clock time: ${wall_time}s <<<"
echo ""
echo "Sequential would have taken ~${sum}s"
echo "Speedup: $(python3 -c "print(round($sum / $wall_time, 1))")x"

rm -rf "$TMPDIR"
