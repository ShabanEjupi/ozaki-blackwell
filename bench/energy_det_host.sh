#!/bin/bash
# energy_det_host.sh N REPS SECONDS -- from the IDE: samples the Windows RAPL package power (1 s) for SECONDS on one ssh
# connection while bench/energy_det.sh runs in WSL on another; prints both traces with their epoch marks.
set -euo pipefail
cd "$(dirname "$0")/.."
n=${1:-2048}; reps=${2:-20}; secs=${3:-80}
PS='$c = Get-Counter "\Energy Meter(RAPL_Package0_PKG)\Power" -SampleInterval 1 -MaxSamples '$secs'; foreach ($s in $c.CounterSamples) { "{0} {1}" -f ([DateTimeOffset]$s.Timestamp).ToUnixTimeMilliseconds(), $s.CookedValue }'
ENC=$(printf '%s' "$PS" | iconv -t UTF-16LE | base64 -w0)
ssh ampere "timeout $((secs + 30)) ssh -o BatchMode=yes -i ~/.ssh/laptop_ed25519 -p 16222 NB@127.0.0.1 'powershell -NoProfile -EncodedCommand $ENC'" \
  | tr -d '\r' > out_rapl.txt &
TIMEOUT=$((secs + 60)) ./wsl.sh "bash bench/energy_det.sh $n $reps" > out_energy.txt
wait
echo "RAPL_BEGIN"; cat out_rapl.txt; echo "RAPL_END"; cat out_energy.txt
