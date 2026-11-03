#!/bin/bash
# hold.sh <tag> <command...> -- runs <command> with the laptop to itself: takes /opt/ozaki/spex.want (the training yields
# between its 2-step runs; night_checks.sh's SPEX and the voice job wait in turn), waits for AC, for the running training run
# to end and for any SPEX run, runs the command, gives the machine back. Sourced (". llm/hold.sh"), it only defines hold().
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
ac() { [ "$($PS -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d "\r ")" != 1 ]; }
hold() {   # hold <tag>: returns once the machine is ours; spex.want goes when the calling shell exits
    # noclobber: of two holders that see the marker gone in the same minute, one creates it and the other waits on
    until ( set -C; echo "$1 <$$>" > /opt/ozaki/spex.want ) 2> /dev/null; do sleep 60; done; trap 'rm -f /opt/ozaki/spex.want' EXIT
    while ! ac || pgrep -x out_jlora > /dev/null || pgrep -x out_spex > /dev/null || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    tag=${1:?tag}; shift; hold "$tag"; echo "$(date '+%F %T') hold $tag: machine held, running: $*"; "$@"; rc=$?
    echo "$(date '+%F %T') hold $tag: rc=$rc, machine given back"; exit $rc
fi
