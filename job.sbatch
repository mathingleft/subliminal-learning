#!/bin/bash
# Generic job script: $1 = code directory, the rest = arguments to the venv python.
set -euo pipefail
cd "$1"
shift
exec /scratch/users/jkli/subliminal-learning/.venv/bin/python "$@"
