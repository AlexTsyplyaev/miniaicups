#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
agent_folder=$(realpath "$script_dir/../players")
runner_folder=$(realpath "${RUNNER}")
num_runs=10
if [[ ! -z "${1}"  ]]
then
    num_runs="${1}"
fi

echo "------------------------------------"
echo "agent location: $agent_folder"
echo "runner location: $runner_folder"
echo "sessions number: $num_runs"
echo "------------------------------------"
echo "Game:"

for ((i=0; i < $num_runs; ++i));
do
    python3 -u $runner_folder/localrunner.py -f "python3 -u $agent_folder/pytorch_main.py --train" -s "python3 -u $agent_folder/pytorch_main.py --train"
done
