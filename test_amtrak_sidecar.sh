#!/bin/bash
# Timing test: Amtrak routine through sidecar with rotating proxies (host:port:user:pass)
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
  "68.182.105.195:62206:buykvgtw:t669WYCuRalP"
  "68.182.109.24:53270:buykvgtw:t669WYCuRalP"
  "68.182.105.233:63284:buykvgtw:t669WYCuRalP"
  "68.182.110.103:52150:buykvgtw:t669WYCuRalP"
  "68.182.104.73:55340:buykvgtw:t669WYCuRalP"
)

SIDECAR_PORT=9223
ROUTINE="example_data/example_routines/amtrak_one_way_train_search_routine.json"
PARAMS='{"origin": "BOS", "destination": "NYP", "departureDate": "2026-08-22"}'

# Check sidecar is running
if ! curl -s "http://127.0.0.1:${SIDECAR_PORT}/json/version" > /dev/null 2>&1; then
  echo "ERROR: Sidecar not running on port ${SIDECAR_PORT}"
  echo "Start it with: python cdp_proxy_sidecar.py --listen-port ${SIDECAR_PORT} --chrome-port 9222"
  exit 1
fi
echo "Sidecar detected on port ${SIDECAR_PORT}"

total=0
times=()

echo ""
echo "Amtrak BOS → NYP (2026-08-22) via SIDECAR — 10 proxies"
echo "------------------------------------------------------------------------"
printf "%-5s %-3s %7s  %s\n" "Run" "OK" "Time" "Proxy"
echo "------------------------------------------------------------------------"

i=0
for entry in "${PROXIES[@]}"; do
  ((i++))

  # Parse host:port:user:pass → user:pass@host:port (sidecar format)
  IFS=':' read -r host port user pass <<< "$entry"
  addr="${user}:${pass}@${host}:${port}"
  label="${host}:${port}"

  start=$(python3 -c 'import time; print(time.time())')

  output=$(bluebox-execute \
    --routine-path "$ROUTINE" \
    --parameters-dict "$PARAMS" \
    --proxy-address "$addr" \
    --remote-debugging-address "http://127.0.0.1:${SIDECAR_PORT}" \
    2>&1)

  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(round($end - $start, 2))")

  ok=$(echo "$output" | grep "ok=True" | tail -1)
  if [ -n "$ok" ]; then
    status="✅"
  else
    status="❌"
  fi

  printf "%-5s %s  %6ss  %s\n" "$i" "$status" "$elapsed" "$label"

  times+=("$elapsed")
  total=$(python3 -c "print(round($total + $elapsed, 2))")
done

avg=$(python3 -c "print(round($total / ${#PROXIES[@]}, 2))")
min=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
max=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)

echo "------------------------------------------------------------------------"
echo "Total time:     ${total}s"
echo "Average time:   ${avg}s"
echo "Min time:       ${min}s"
echo "Max time:       ${max}s"
echo ""
echo "Auth: handled by sidecar (no Fetch interception, 0 paused requests)"
