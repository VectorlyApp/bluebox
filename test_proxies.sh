#!/bin/bash
# Test all proxy endpoints with the get_ip_address routine

PROXIES=(
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

passed=0
failed=0

for proxy in "${PROXIES[@]}"; do
  addr="http://${USER}:${PASS}@${proxy}"

  printf "%-22s → " "$proxy"
  output=$(bluebox-execute --routine-path "$ROUTINE" --parameters-dict '{}' --proxy-address "$addr" 2>&1)

  # Extract IP from data='x.x.x.x' in the log output
  ip=$(echo "$output" | sed -n "s/.*data='\([0-9.]*\)'.*/\1/p" | tail -1)

  if [ -n "$ip" ]; then
    printf "✅ %s\n" "$ip"
    ((passed++))
  else
    printf "❌ FAILED\n"
    ((failed++))
  fi
done

echo ""
echo "Results: $passed passed, $failed failed out of ${#PROXIES[@]} proxies"
