#!/bin/bash
# Timing test: run get_ip_address routine through 10 different proxies

PROXIES=(
  "208.66.76.226:6150"
  "207.228.29.185:5676"
  "72.46.138.9:6235"
  "103.210.12.42:5970"
  "216.98.255.235:6857"
  "130.180.233.147:7718"
  "63.246.130.97:6298"
  "45.58.244.167:6580"
  "64.52.28.228:7915"
  "9.142.43.181:5351"
)

USER="enjakyjo"
PASS="cs1vy9f77jq0"
ROUTINE="example_data/example_routines/get_ip_address_routine.json"

total=0
total_auth=0
total_paused=0
times=()

echo "Running 10 iterations across 10 proxies..."
echo "------------------------------------------------------------------------"
printf "%-5s %-3s %-16s %7s  %5s %7s  %s\n" "Run" "OK" "IP" "Time" "Auth" "Paused" "Proxy"
echo "------------------------------------------------------------------------"

i=0
for proxy in "${PROXIES[@]}"; do
  ((i++))
  addr="http://${USER}:${PASS}@${proxy}"

  start=$(python3 -c 'import time; print(time.time())')

  output=$(bluebox-execute --routine-path "$ROUTINE" --parameters-dict '{}' --proxy-address "$addr" 2>&1)

  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(round($end - $start, 2))")

  ip=$(echo "$output" | sed -n "s/.*data='\([0-9.]*\)'.*/\1/p" | tail -1)
  status="✅"
  [ -z "$ip" ] && status="❌" && ip="FAILED"

  # Extract proxy auth stats from log
  auth=$(echo "$output" | sed -n 's/.*\([0-9]*\) auth challenges handled.*/\1/p' | tail -1)
  paused=$(echo "$output" | sed -n 's/.*handled, \([0-9]*\) paused.*/\1/p' | tail -1)
  [ -z "$auth" ] && auth=0
  [ -z "$paused" ] && paused=0

  printf "%-5s %s  %-16s %6ss  %5s %7s  %s\n" "$i" "$status" "$ip" "$elapsed" "$auth" "$paused" "$proxy"

  times+=("$elapsed")
  total=$(python3 -c "print(round($total + $elapsed, 2))")
  total_auth=$((total_auth + auth))
  total_paused=$((total_paused + paused))
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
echo "Total auth challenges handled:    $total_auth"
echo "Total paused requests continued:  $total_paused"
