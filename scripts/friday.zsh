#!/usr/bin/env zsh

set -euo pipefail

friday() {
    exec uv run python ~/proj/friday-cli/friday/main.py "$@"
    if [ -f ~/.friday/exec_cmd ]; then
        command=$(cat ~/.friday/exec_cmd)
        rm ~/.friday/exec_cmd
        print -z -- "$command"
    fi
}
