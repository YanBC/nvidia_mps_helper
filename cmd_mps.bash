#!/bin/bash

if [ $GPU_ID = "" ];
then
    echo "No GPU ID specified"
    exit 2
fi

# check number of arguments
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

# env variables
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$GPU_ID

echo $1 | nvidia-cuda-mps-control
