#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
pushd  $SCRIPT_DIR &> /dev/null
poetry build
./language_container/exaslct export --flavor-path language_container/exasol-data-science-utils-python-container/
