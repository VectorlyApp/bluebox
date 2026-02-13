#!/bin/bash
# Timing test: Amtrak routine through 10 rotating proxies (host:port:user:pass format)

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

ROUTINE="example_data/example_routines/amtrak_one_way_train_search_routine.json"
PARAMS='{"origin": "BOS", "destination": "NYP", "departureDate": "2026-08-22"}'

total=0
total_auth=0
total_paused=0
times=()

echo "Amtrak BOS → NYP (2026-08-22) through 10 proxies"
echo "------------------------------------------------------------------------"
printf "%-5s %-3s %7s  %5s %7s  %s\n" "Run" "OK" "Time" "Auth" "Paused" "Proxy"
echo "------------------------------------------------------------------------"

i=0
for entry in "${PROXIES[@]}"; do
  ((i++))

  # Parse host:port:user:pass
  IFS=':' read -r host port user pass <<< "$entry"
  addr="http://${user}:${pass}@${host}:${port}"
  label="${host}:${port}"

  start=$(python3 -c 'import time; print(time.time())')

  output=$(bluebox-execute --routine-path "$ROUTINE" --parameters-dict "$PARAMS" --proxy-address "$addr" 2>&1)

  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(round($end - $start, 2))")

  ok=$(echo "$output" | grep "ok=True" | tail -1)
  if [ -n "$ok" ]; then
    status="✅"
  else
    status="❌"
  fi

  auth=$(echo "$output" | sed -n 's/.*\([0-9]*\) auth challenges handled.*/\1/p' | tail -1)
  paused=$(echo "$output" | sed -n 's/.*handled, \([0-9]*\) paused.*/\1/p' | tail -1)
  [ -z "$auth" ] && auth=0
  [ -z "$paused" ] && paused=0

  printf "%-5s %s  %6ss  %5s %7s  %s\n" "$i" "$status" "$elapsed" "$auth" "$paused" "$label"

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
