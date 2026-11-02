# deploy_bin.sh -- sourced by the *_check.sh scripts (needs their log()):
#   deploy_bin <new binary> <label>
# puts <new binary> in place of /opt/ozaki/src/out_jlora with an atomic mv (a running training keeps its inode) and keeps the
# binary it replaces as out_jlora.pre-<label> -- only the first time. 31-10: rows_check.sh ran twice; the second run copied
# the binary the first had just deployed to out_jlora.pre-rows, and the real previous one (b49fbd3) was gone. A binary that is
# already in place is not deployed again.
deploy_bin() {
    local R=${DEPLOY_R:-/opt/ozaki/src} new=$1 lb=$2
    if cmp -s "$new" $R/out_jlora; then log "already deployed ($lb): $R/out_jlora is this binary"; return 0; fi
    [ -e $R/out_jlora.pre-$lb ] || cp $R/out_jlora $R/out_jlora.pre-$lb
    cp "$new" $R/out_jlora.$lb.tmp && mv -f $R/out_jlora.$lb.tmp $R/out_jlora &&
        log "deployed: $R/out_jlora = $(cat .version 2>/dev/null | tr '\n' ' ') (old: out_jlora.pre-$lb)"
}
