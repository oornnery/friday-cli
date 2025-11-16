#!/usr/bin/env zsh

set -euo pipefail

exec uv run python ~/proj/friday-cli/src/main.py "$@"
