#!/usr/bin/env bash
#PBS -l select=1:ncpus=2:ngpus=1:cl_gram=False:cl_konos=False:mem=32gb
#PBS -l walltime=24:00:00
#PBS -q gpu

module add python-3.6.2-gcc

home_dir="/storage/plzen1/home/benesjan"
timestamp=$(date +%s)

source $home_dir/spc/venv/bin/activate

export TMPDIR=$home_dir/.tmp
export PYTHONPATH="${PYTHONPATH}:$home_dir/spc"

module add pytorch-1.1.0_python-3.6.2_cuda-10.1
module add opencv-3.4.5-py36-cuda10.1

python "$home_dir/spc/$SCRIPT_NAME.py"