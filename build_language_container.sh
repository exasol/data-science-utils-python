#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
pushd  $SCRIPT_DIR &> /dev/null
poetry build

RELEASE_BUILD_STEP_DIST_DIRECTORY="language_container/exasol-data-science-utils-python-container/flavor_base/release/dist"
echo "Copy" dist/*.whl "$RELEASE_BUILD_STEP_DIST_DIRECTORY"
mkdir "$RELEASE_BUILD_STEP_DIST_DIRECTORY" || true
cp dist/*.whl "$RELEASE_BUILD_STEP_DIST_DIRECTORY"
trap 'rm -rf "$RELEASE_BUILD_STEP_DIST_DIRECTORY"' EXIT

echo "Build container"
./language_container/exaslct export --flavor-path language_container/exasol-data-science-utils-python-container/
