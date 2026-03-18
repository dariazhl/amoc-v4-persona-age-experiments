#!/bin/bash
set -euo pipefail

SCRIPT="$1"
shift

python "$SCRIPT" "$@"
