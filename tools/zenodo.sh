#!/bin/bash
# zenodo.sh <paper dir> <license id> [publish] -- a Zenodo record for one paper of this repository, run on ampere (curl, jq):
# a new deposition, <dir>/main.pdf, the metadata of <dir>/zenodo.json with the license added (e.g. cc-by-4.0,
# cc-by-nc-nd-4.0: the author chooses), and the record's link. Without "publish" the record stays a private draft that only
# the account sees; "publish" makes it public and permanent (a DOI cannot be withdrawn) -- only after the author's yes to the
# full text. The token is read by its label (ZENODO_TOKEN) from the credentials file, never printed.
set -euo pipefail
D=${1:?paper dir}; LIC=${2:?license id}; PUB=${3:-}
F=$(echo /mnt/data/workspace/spacecode-brain/credentials.local.txt)
T=$(grep -oP '^ZENODO_TOKEN=\K\S+' "$F"); A="Authorization: Bearer $T"; Z=https://zenodo.org/api
[ -s "$D/main.pdf" ] && [ -s "$D/zenodo.json" ] || { echo "need $D/main.pdf and $D/zenodo.json"; exit 1; }
dep=$(curl -fsS -H "$A" -H "Content-Type: application/json" -X POST "$Z/deposit/depositions" -d '{}')
id=$(jq -r .id <<< "$dep"); bucket=$(jq -r .links.bucket <<< "$dep")
curl -fsS -H "$A" --upload-file "$D/main.pdf" "$bucket/$(basename "$D").pdf" > /dev/null
jq --arg l "$LIC" '.metadata.license = $l' "$D/zenodo.json" | curl -fsS -H "$A" -H "Content-Type: application/json" -X PUT "$Z/deposit/depositions/$id" -d @- > /dev/null
echo "draft $id: https://zenodo.org/uploads/$id"
if [ "$PUB" = publish ]; then
    r=$(curl -fsS -H "$A" -X POST "$Z/deposit/depositions/$id/actions/publish")
    echo "published: DOI $(jq -r .doi <<< "$r"), $(jq -r .links.record_html <<< "$r")"
fi
