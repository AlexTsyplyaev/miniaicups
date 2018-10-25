#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
runner=$(realpath "$script_dir/run_original_env.sh")
train="--train"
runs=""

num="${1}"
i=0
for ((i=0;i<num;++i)); do
    ${runner} ${@:2}
done
