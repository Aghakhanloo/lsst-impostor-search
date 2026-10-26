#!/bin/bash
# ^ This file is a bash script: a list of terminal commands run one after another.
#
# Download the public ORACLE-1 "lite" weights (Shah et al. 2025, https://doi.org/10.3847/1538-4357/ae1130;
# github.com/dev-ved30/Oracle, modified MIT license with a non-military-use restriction)
# and check their checksum. The weights are what ORACLE learned in training.

# Stop immediately if any command fails.
set -e

# Where to save the weights: ml/pretrained/ (.pth is PyTorch's format for a trained model).
# $0 is this script's path; dirname gives its folder (ml/).
DEST="$(dirname "$0")/pretrained/oracle1_elasticc_lite.pth"
# A fixed version (commit) of ORACLE's GitHub repo, so we always get the same weights.
COMMIT=9f7668393113b038dd9aaf4b454ab9acfa7593db
# Web address of the weights file at that version.
URL="https://raw.githubusercontent.com/dev-ved30/Oracle/${COMMIT}/models/ELAsTiCC-lite/revived-star-159/best_model_f1.pth"
# Expected checksum: a "fingerprint" of the file; any change to the file changes it.
SHA=46f915c48b03bc6324435873bad9e249fdc9f242f8974472d9cd55d9e560b4e2

# Make the ml/pretrained/ folder (-p: no complaint if it already exists).
mkdir -p "$(dirname "$DEST")"
# Download only if the file does not exist yet ("!" means "not"; "fi" ends the if).
if [ ! -f "$DEST" ]; then
    echo "Downloading ORACLE-1 weights..."
    curl -sSL -o "$DEST" "$URL"
fi
# Compare the file's fingerprint with the expected one: prints OK, or fails if the file is wrong.
echo "${SHA}  ${DEST}" | sha256sum -c -
