# gatt_test.sh -- after the CTI job: the first CTI-MCQ chunk (prompts 0-154) with the attention on the GPU, OZ_GATT=2 (GPU and
# host, every output compared) then OZ_GATT=1 (GPU only, the time); the score lines must equal those of the host run.
# SPEX waits on /opt/ozaki/test.busy, refreshed until gatt.done.
B=/opt/ozaki/b; C=/opt/ozaki/cti; J=/opt/ozaki/jais; cd $B
(while [ ! -e $B/gatt.done ]; do touch /opt/ozaki/test.busy; sleep 600; done) &
until [ -e $B/job.done ] && ! pgrep -x out_jais_batch2 > /dev/null; do sleep 60; done
head -155 $C/mcq_ap.ids > gatt_ids.txt; grep "^score" cti_ap.txt | head -155 > gatt_ref.txt
for m in 2 1; do
    echo "$(date '+%F %T') mode $m start" >> gatt.log
    OZ_GATT=$m OZ_AHEAD=3 OZ_PERPROMPT=1 OZ_SCORE=358,408,353,421 ./out_jais_batch3 $J gatt_ids.txt 0 > gatt$m.txt 2> gatt$m.err
    echo "$(date '+%F %T') mode $m rc=$? $(grep '^pass' gatt$m.err | tail -1)" >> gatt.log
    grep "^score" gatt$m.txt | cmp - gatt_ref.txt >> gatt.log 2>&1 && echo "mode $m: 155 score lines identical to the host run" >> gatt.log
    grep "^batched" gatt$m.txt >> gatt.log
done
touch gatt.done; rm -f /opt/ozaki/test.busy
