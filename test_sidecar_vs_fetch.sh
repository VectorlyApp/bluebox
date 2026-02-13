#!/bin/bash
# Head-to-head: Sidecar vs Fetch-based proxy auth — Mass Corp Search
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
SIDECAR_ADDR="http://127.0.0.1:${SIDECAR_PORT}"
CHROME_ADDR="http://127.0.0.1:9222"
ROUTINE="example_data/example_routines/massachusetts_corp_search_routine.json"
PARAMS='{"entity_name": "Microsoft"}'

# Check sidecar is running
if ! curl -s "http://127.0.0.1:${SIDECAR_PORT}/json/version" > /dev/null 2>&1; then
  echo "ERROR: Sidecar not running on port ${SIDECAR_PORT}"
  echo "Start it with: python cdp_proxy_sidecar.py --listen-port ${SIDECAR_PORT} --chrome-port 9222"
  exit 1
fi
echo "Sidecar detected on port ${SIDECAR_PORT}"
echo ""

run_method() {
  local method_name="$1"
  local extra_flags="$2"
  local count=${#PROXIES[@]}

  local total=0
  local times=()
  local passed=0
  local failed=0
  local total_auth=0
  local total_paused=0

  echo "========================================================================"
  echo " ${method_name} — Mass Corp Search (Microsoft) — ${count} proxies"
  echo "========================================================================"
  printf "%-5s %-3s %7s  %5s %7s  %s\n" "Run" "OK" "Time" "Auth" "Paused" "Proxy"
  echo "------------------------------------------------------------------------"

  local i=0
  for entry in "${PROXIES[@]}"; do
    ((i++))
    IFS=':' read -r host port user pass <<< "$entry"
    local addr="http://${user}:${pass}@${host}:${port}"
    local label="${host}:${port}"

    # Use user:pass@host:port format (no scheme)
    addr="${user}:${pass}@${host}:${port}"

    local start=$(python3 -c 'import time; print(time.time())')

    local output=$(bluebox-execute \
      --routine-path "$ROUTINE" \
      --parameters-dict "$PARAMS" \
      --proxy-address "$addr" \
      $extra_flags \
      2>&1)

    local end=$(python3 -c 'import time; print(time.time())')
    local elapsed=$(python3 -c "print(round($end - $start, 2))")

    local ok=$(echo "$output" | grep "ok=True" | tail -1)
    local status="❌"
    [ -n "$ok" ] && status="✅" && ((passed++)) || ((failed++))

    local auth=$(echo "$output" | sed -n 's/.*\([0-9]*\) auth challenges handled.*/\1/p' | tail -1)
    local paused=$(echo "$output" | sed -n 's/.*handled, \([0-9]*\) paused.*/\1/p' | tail -1)
    [ -z "$auth" ] && auth=0
    [ -z "$paused" ] && paused=0

    printf "%-5s %s  %6ss  %5s %7s  %s\n" "$i" "$status" "$elapsed" "$auth" "$paused" "$label"

    times+=("$elapsed")
    total=$(python3 -c "print(round($total + $elapsed, 2))")
    total_auth=$((total_auth + auth))
    total_paused=$((total_paused + paused))
  done

  local avg=$(python3 -c "print(round($total / $count, 2))")
  local min=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
  local max=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)

  echo "------------------------------------------------------------------------"
  echo "Passed: $passed  Failed: $failed"
  echo "Total: ${total}s  Avg: ${avg}s  Min: ${min}s  Max: ${max}s"
  echo "Auth challenges: $total_auth  Paused requests: $total_paused"
  echo ""
}

# ── Method 1: Sidecar ──
run_method "SIDECAR" "--remote-debugging-address $SIDECAR_ADDR"

# ── Method 2: Fetch-based auth (direct to Chrome) ──
run_method "FETCH AUTH" ""

# ── Summary ──
echo "========================================================================"
echo " Compare the Avg times above to see the overhead difference."
echo " Sidecar: 0 paused requests (auth at TCP level)"
echo " Fetch:   many paused requests (auth via CDP Fetch domain)"
echo "========================================================================"
