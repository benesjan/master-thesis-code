#!/usr/bin/env bash
#PBS -l select=1:ncpus=2:ngpus=1:cl_gram=False:cl_konos=False:mem=32gb
#PBS -l walltime=24:00:00
#PBS -q gpu

module add python-3.6.2-gcc
module add pytorch-1.1.0_python-3.6.2_cuda-10.1
module add opencv-3.4.5-py36-cuda10.1

python "/storage/plzen1/home/benesjan/spc/$SCRIPT_NAME.py"