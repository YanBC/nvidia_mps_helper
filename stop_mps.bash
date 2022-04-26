#!/bin/bash

if [ $GPU_ID = "" ];
then
    echo "No GPU ID specified"
    exit 2
fi

# env variables
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$GPU_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU_ID

# stop and clean up
echo "quit" | nvidia-cuda-mps-control
