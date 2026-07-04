#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-Automation/automation_config.yaml}"
COMMAND="${2:-}"

if [[ -n "$COMMAND" ]]; then
  python Automation/Inference_stack.py --config "$CONFIG" --command "$COMMAND"
else
  python Automation/Inference_stack.py --config "$CONFIG"
fi
