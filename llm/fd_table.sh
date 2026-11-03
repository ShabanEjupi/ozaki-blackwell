#!/bin/bash
# fd_table.sh <dir> ... -- the finite-difference table of fd_check.sh runs (each <dir> has step.log with the predictions and
# fd.log with the measured losses), the runs pooled: per seed and eps the predicted L(+e) - L(-e) = <g_0, D+ - D->, the
# measured one and their ratio; for the gradient's own direction (seed 0: sign(g_0)) the least-squares line ratio = a + b eps
# over eps >= $EMIN (default 3.5e-5; below it the signal is under the noise), and for the random seeds the measured change
# itself -- the noise floor at that eps, since their predicted change is ~100x smaller.
#   llm/fd_table.sh /opt/ozaki/jlora/fd61 /opt/ozaki/jlora/fd61b
awk -v emin="${EMIN:-3.5e-5}" '
  FILENAME ~ /step\.log$/ && /^predicted seed/ { s = $3; e = $4; sub(/^L\(\+/, "", e); sub(/\)$/, "", e); P[s " " e + 0] = $8 }
  FILENAME ~ /fd\.log$/ && /^seed .* L\+ .* L- / { D[$2 " " $4 + 0] = $6 - $8 }
  END {
    for (k in D) if (k in P) { split(k, b, " "); printf "seed %s eps %-8g predicted %+.6e measured %+.6e ratio %.4f\n", b[1], b[2], P[k], D[k], D[k] / P[k] | "sort -k2,2n -k4,4g" }
    close("sort -k2,2n -k4,4g")
    n = 0; for (k in D) if (k in P) { split(k, b, " "); if (b[1] == 0 && b[2] >= emin) { n++; x = b[2]; y = D[k] / P[k]; sx += x; sy += y; sxx += x * x; sxy += x * y } }
    if (n > 1) { bb = (n * sxy - sx * sy) / (n * sxx - sx * sx); a = (sy - bb * sx) / n
      w = 0; for (k in D) if (k in P) { split(k, c, " "); if (c[1] == 0 && c[2] >= emin) { r = D[k] / P[k] - a - bb * c[2]; if (r < 0) r = -r; if (r > w) w = r } }
      printf "sign(g_0), %d eps >= %g: ratio = %.4f %+.1f eps, largest residual %.4f\n", n, emin, a, bb, w }
    for (k in D) if (k in P) { split(k, c, " "); if (c[1] != 0) { m = D[k] < 0 ? -D[k] : D[k]; printf "noise floor: seed %s eps %g |measured| %.3e (predicted %.3e)\n", c[1], c[2], m, P[k] } }
  }' $(for d in "$@"; do echo "$d/step.log" "$d/fd.log"; done)
