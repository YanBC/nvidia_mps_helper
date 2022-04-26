#!/bin/bash

if [ $GPU_ID = "" ];
then
    echo "No GPU ID specified"
    exit 2
fi

# env variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$GPU_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU_ID

# start mps control daemon
sudo nvidia-smi -i $GPU_ID -c 3 && \
mkdir -p $CUDA_MPS_PIPE_DIRECTORY && \
mkdir -p $CUDA_MPS_LOG_DIRECTORY && \
nvidia-cuda-mps-control -d
