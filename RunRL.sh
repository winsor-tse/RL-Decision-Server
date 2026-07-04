#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-Automation/automation_config.yaml}"
python Automation/Training_stack.py --config "$CONFIG"
