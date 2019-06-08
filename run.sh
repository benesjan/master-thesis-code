#!/usr/bin/env bash

module add python-3.6.2-gcc
module add pytorch-1.1.0_python-3.6.2_cuda-10.1
module add opencv-3.4.5-py36-cuda10.1

SCRIPT_NAME=$1

python "/storage/plzen1/home/benesjan/spc/$SCRIPT_NAME.py"