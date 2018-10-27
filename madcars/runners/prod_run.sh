#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
runner=$(realpath "$script_dir/run_original_env.sh")
export RUNNER=$HOME/projects/orig_miniaicups/madcars/Runners/
${runner}
export RUNNER=""
