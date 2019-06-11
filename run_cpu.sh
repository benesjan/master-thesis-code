#!/usr/bin/env bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=120:00:00

module add python-3.6.2-gcc
module add python36-modules-gcc
module add pytorch-1.1.0_python-3.6.2_cuda-10.1
module add opencv-3.4.5-py36-cuda10.1

home_dir="/storage/plzen1/home/benesjan"
timestamp=$(date +%s)

python "$home_dir/spc/$SCRIPT_NAME.py" > "$home_dir/output_cpu_$timestamp.txt"