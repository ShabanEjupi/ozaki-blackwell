# vf-offset.ps1 [-Set <MHz>] -- reads (and optionally sets) the GPC clock V/F-curve offset through NVIDIA's own nvml.dll.
# A positive offset gives more MHz at the same voltage: under the laptop's fixed 95 W cap that is more TOPS per watt
# (the effect of an undervolt). It does not persist: a reboot or driver reload restores 0. Needs an elevated session.
param([int] $Set = [int]::MinValue)
Add-Type -Namespace Nv -Name M -MemberDefinition @'
[DllImport("nvml.dll")] public static extern int nvmlInit_v2();
[DllImport("nvml.dll")] public static extern int nvmlDeviceGetHandleByIndex_v2(uint i, out IntPtr d);
[DllImport("nvml.dll")] public static extern int nvmlDeviceGetGpcClkMinMaxVfOffset(IntPtr d, out int lo, out int hi);
[DllImport("nvml.dll")] public static extern int nvmlDeviceGetGpcClkVfOffset(IntPtr d, out int o);
[DllImport("nvml.dll")] public static extern int nvmlDeviceSetGpcClkVfOffset(IntPtr d, int o);
[DllImport("nvml.dll")] public static extern int nvmlDeviceGetMemClkMinMaxVfOffset(IntPtr d, out int lo, out int hi);
'@
$r = [Nv.M]::nvmlInit_v2(); if ($r) { "nvmlInit $r"; exit 1 }
$d = [IntPtr]::Zero; $r = [Nv.M]::nvmlDeviceGetHandleByIndex_v2(0, [ref]$d); if ($r) { "handle $r"; exit 1 }
$lo = 0; $hi = 0; $o = 0
$r = [Nv.M]::nvmlDeviceGetGpcClkMinMaxVfOffset($d, [ref]$lo, [ref]$hi); "gpc offset range: rc=$r [$lo, $hi] MHz"
$r = [Nv.M]::nvmlDeviceGetMemClkMinMaxVfOffset($d, [ref]$lo, [ref]$hi); "mem offset range: rc=$r [$lo, $hi] MHz"
$r = [Nv.M]::nvmlDeviceGetGpcClkVfOffset($d, [ref]$o); "gpc offset now: rc=$r $o MHz"
if ($Set -ne [int]::MinValue) {
    $r = [Nv.M]::nvmlDeviceSetGpcClkVfOffset($d, $Set); "set $Set MHz: rc=$r"
    $r = [Nv.M]::nvmlDeviceGetGpcClkVfOffset($d, [ref]$o); "gpc offset now: rc=$r $o MHz"
}
