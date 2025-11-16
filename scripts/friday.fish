#!/usr/bin/env fish

function friday
    uv run python ~/proj/friday-cli/friday/main.py $argv
    if test -f ~/.friday/exec_cmd
        set cmd (cat ~/.friday/exec_cmd)
        rm ~/.friday/exec_cmd
        commandline -r "$cmd"
    end
end
