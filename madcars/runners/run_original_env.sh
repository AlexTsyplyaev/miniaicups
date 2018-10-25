#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
agent_folder=$(realpath "$script_dir/../players")
runner_folder=$(realpath "$script_dir")
train="--train"
runs=""

if [[ ! -z "${RUNNER}"  ]]
then
    runner_folder=$(realpath "${RUNNER}")
    train=""
fi

if [[ ! -z "${1}"  ]]
then
    runs="-g ${1}"
fi

full=""
if [[ ! -z "${2}"  ]]
then
    full="--full"
fi

echo "------------------------------------"
echo "agent location: $agent_folder"
echo "runner location: $runner_folder"
echo "options: $runs $full $train"
echo "------------------------------------"
echo "Game:"

python3 -u $runner_folder/localrunner.py -f "python3 -u $agent_folder/pytorch_main.py $train" -s "python3 -u $agent_folder/pytorch_main.py $train" ${runs} ${full}
