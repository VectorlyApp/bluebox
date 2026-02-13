#!/bin/bash
# Timing test: Mass Corp Search via sidecar — two proxy sets
#
# Prerequisites:
#   1. Chrome running on port 9222
#   2. Sidecar running: python cdp_proxy_sidecar.py --listen-port 9223 --chrome-port 9222

# Set 1: Rotating residential (host:port:user:pass)
SET1_NAME="Rotating Residential"
SET1_PROXIES=(
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

# Set 2: Static residential (host:port:user:pass)
SET2_NAME="Static Residential"
SET2_PROXIES=(
  "208.66.76.226:6150:enjakyjo:cs1vy9f77jq0"
  "207.228.29.185:5676:enjakyjo:cs1vy9f77jq0"
  "72.46.138.9:6235:enjakyjo:cs1vy9f77jq0"
  "103.210.12.42:5970:enjakyjo:cs1vy9f77jq0"
  "216.98.255.235:6857:enjakyjo:cs1vy9f77jq0"
  "130.180.233.147:7718:enjakyjo:cs1vy9f77jq0"
  "63.246.130.97:6298:enjakyjo:cs1vy9f77jq0"
  "45.58.244.167:6580:enjakyjo:cs1vy9f77jq0"
  "64.52.28.228:7915:enjakyjo:cs1vy9f77jq0"
  "9.142.43.181:5351:enjakyjo:cs1vy9f77jq0"
)

SIDECAR_PORT=9223
SIDECAR_ADDR="http://127.0.0.1:${SIDECAR_PORT}"
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

run_set() {
  local set_name="$1"
  shift
  local proxies=("$@")
  local count=${#proxies[@]}

  local total=0
  local times=()
  local passed=0
  local failed=0

  echo "========================================================================"
  echo " Mass Corp Search (Microsoft) — ${set_name} — ${count} proxies"
  echo "========================================================================"
  printf "%-5s %-3s %7s  %s\n" "Run" "OK" "Time" "Proxy"
  echo "------------------------------------------------------------------------"

  local i=0
  for entry in "${proxies[@]}"; do
    ((i++))
    IFS=':' read -r host port user pass <<< "$entry"
    local addr="${user}:${pass}@${host}:${port}"
    local label="${host}:${port}"

    local start=$(python3 -c 'import time; print(time.time())')

    local output=$(bluebox-execute \
      --routine-path "$ROUTINE" \
      --parameters-dict "$PARAMS" \
      --proxy-address "$addr" \
      --remote-debugging-address "$SIDECAR_ADDR" \
      2>&1)

    local end=$(python3 -c 'import time; print(time.time())')
    local elapsed=$(python3 -c "print(round($end - $start, 2))")

    local ok=$(echo "$output" | grep "ok=True" | tail -1)
    local status="❌"
    [ -n "$ok" ] && status="✅" && ((passed++)) || ((failed++))

    printf "%-5s %s  %6ss  %s\n" "$i" "$status" "$elapsed" "$label"

    times+=("$elapsed")
    total=$(python3 -c "print(round($total + $elapsed, 2))")
  done

  local avg=$(python3 -c "print(round($total / $count, 2))")
  local min=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
  local max=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)

  echo "------------------------------------------------------------------------"
  echo "Passed: $passed  Failed: $failed"
  echo "Total: ${total}s  Avg: ${avg}s  Min: ${min}s  Max: ${max}s"
  echo ""
}

run_set "$SET1_NAME" "${SET1_PROXIES[@]}"
run_set "$SET2_NAME" "${SET2_PROXIES[@]}"
