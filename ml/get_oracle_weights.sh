#!/bin/bash
# Download the public ORACLE-1 "lite" weights (Shah et al. 2025; github.com/dev-ved30/Oracle,
# modified MIT license with a non-military-use restriction) and check their checksum.
set -e
DEST="$(dirname "$0")/pretrained/oracle1_elasticc_lite.pth"
COMMIT=9f7668393113b038dd9aaf4b454ab9acfa7593db
URL="https://raw.githubusercontent.com/dev-ved30/Oracle/${COMMIT}/models/ELAsTiCC-lite/revived-star-159/best_model_f1.pth"
SHA=46f915c48b03bc6324435873bad9e249fdc9f242f8974472d9cd55d9e560b4e2

mkdir -p "$(dirname "$DEST")"
if [ ! -f "$DEST" ]; then
    echo "Downloading ORACLE-1 weights..."
    curl -sSL -o "$DEST" "$URL"
fi
echo "${SHA}  ${DEST}" | sha256sum -c -
