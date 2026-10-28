# rcm_run.sh -- after gatt_test.sh: CTI-RCM (1000 CVE -> CWE, cti_prep.py NOTHINK=1 PREFIX="CWE-", 5 new tokens per prompt),
# the attention on the GPU only if the GPU/host comparison found 0 differing outputs and the same 155 score lines; then SPEX
# (it waits on /opt/ozaki/test.busy, refreshed until rcm.done).
B=/opt/ozaki/b; C=/opt/ozaki/cti; J=/opt/ozaki/jais; cd $B
(while [ ! -e $B/rcm.done ]; do touch /opt/ozaki/test.busy; sleep 600; done) &
until [ -e $B/gatt.done ]; do sleep 60; done
G=0; grep -q "mode 2: 155 score lines identical" gatt.log && grep -q "differing GPU/host 0$" gatt.log && G=1
echo "$(date '+%F %T') RCM start, OZ_GATT=$G" >> gatt.log
OZ_GATT=$G OZ_AHEAD=3 OZ_PERPROMPT=1 ./out_jais_batch3 $J $C/rcm_nt.ids 5 > rcm.txt 2> rcm.err
echo "$(date '+%F %T') RCM rc=$? $(grep '^batched' rcm.txt)" >> gatt.log
python3 -I /opt/ozaki/src/llm/cti_rcm_score.py rcm.txt $C/rcm_nt.ids $C/rcm_nt.gt $J >> gatt.log 2>&1
touch rcm.done; sleep 5; rm -f /opt/ozaki/test.busy
