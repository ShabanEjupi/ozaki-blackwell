#!/bin/bash
# win.sh -- sourced by every script that reads or drives Windows from inside WSL (". llm/win.sh"): PS, a live interop socket,
# pw "<command>", ac. Why one file (04-11-2026): a job started from ssh with "setsid -f" outlives the wsl.exe that started it,
# and with it the socket named in $WSL_INTEROP. powershell.exe then fails ("UtilAcceptVsock: accept4 failed 110") and each of
# seven scripts read the failure as an answer: the AC check passed as "on AC" (an empty string is not 1), the RAPL window of
# energy_cpudd.sh had 0 samples and its "12.8 W" was -1 + 13.8, Start/Stop-ScheduledTask did nothing. /run/WSL/1_interop
# (the init's socket) lives as long as the distribution does.
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
win_ok() { [ "$(timeout 60 "$PS" -NoProfile -Command 7 < /dev/null 2> /dev/null | tr -dc 0-9)" = 7 ]; }
win_interop() {   # keeps $WSL_INTEROP if it answers, else exports the first socket that does; returns 1 (and says so) if none
    win_ok && return 0
    local s; for s in /run/WSL/1_interop /run/WSL/*_interop; do
        [ -S "$s" ] || [ -L "$s" ] || continue
        if WSL_INTEROP=$s win_ok; then export WSL_INTEROP=$s; return 0; fi
    done
    echo "[win.sh] $(date '+%F %T') no live WSL interop socket: Windows cannot be read from here" >&2; return 1
}
pw() { timeout 120 "$PS" -NoProfile -Command "$1" < /dev/null 2> /dev/null | tr -d '\r'; }   # stdin closed: powershell.exe reads it
battery() {   # Win32_Battery.BatteryStatus (1 = discharging, 2 = on AC), digits only; empty once more after a fresh socket = unknown
    local b; b=$(pw '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -dc 0-9)
    [ -n "$b" ] || { win_interop > /dev/null 2>&1 && b=$(pw '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -dc 0-9); }
    echo "$b"
}
ac() {   # true unless the machine is known to run on battery; an unreadable state is said on stderr, not taken as AC in silence
    local b; b=$(battery)
    [ -n "$b" ] || echo "[win.sh] $(date '+%F %T') battery state unreadable (taken as AC)" >&2
    [ "$b" != 1 ]
}
win_interop
