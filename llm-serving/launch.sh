#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# If first arg is a sub-command (validate, plan, run, report), pass it directly.
# Otherwise default to 'run'.
if [ $# -gt 0 ] && [[ "$1" =~ ^(validate|plan|run|report)$ ]]; then
    exec uv run python -m llm_serving.cli "$@"
else
    exec uv run python -m llm_serving.cli run "$@"
fi
