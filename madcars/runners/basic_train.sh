#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
runner=$(realpath "$script_dir/multi_run.sh")

${runner} 5 500
${runner} 1 100 --full
${runner} 1 200 --full
${runner} 1 50 --full --no-eps
${runner} 2 200
