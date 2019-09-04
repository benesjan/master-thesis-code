#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=120:00:00

module add python-3.6.2-gcc

home_dir="/storage/plzen1/home/benesjan"
timestamp=$(date +%s)

export TMPDIR=$home_dir/tmp

source $home_dir/spc/venv/bin/activate

python "$home_dir/spc/$SCRIPT_NAME.py"