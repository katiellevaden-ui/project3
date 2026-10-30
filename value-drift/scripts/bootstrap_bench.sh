#!/usr/bin/env bash
# Install the inference bench into the pod's training venv so the bench loop can run its
# Inspect measurement tasks there. Run after bootstrap_gpu.sh / bootstrap_stage2.sh.
#   bash scripts/bootstrap_bench.sh /workspace/venv /workspace/project3
set -euo pipefail
VENV=${1:-/workspace/venv}; ROOT=${2:-$(cd "$(dirname "$0")/../.." && pwd)}
"$VENV/bin/pip" install -e "$ROOT/objective-editing" -e "$ROOT/value-drift-bench" "inspect-ai>=0.3.222" python-dotenv
echo "bench installed into $VENV; set OPENROUTER_API_KEY in value-drift/.env for the judge"
"$VENV/bin/python" -c "import valuedrift, objective_editing, inspect_ai; print('imports ok')"
