## Overview
MPS stands for multi-process service. It enables multiple cuda contexts to run concurrently on the same GPU. See [NVDIA MPS](https://docs.nvidia.com/deploy/mps/index.html) for more info.

Note:
1. Only available on post-Kepler GPUs, i.e. SM 3.5 or later
2. Only available on Linux system (as of April 2022)
3. The MPS control daemon is bound to a specific GPU device so if you have multiple GPUs, you will have to start mutlitple control daemons

## How MPS work?
![How MPS work?](./imgs/how_mps_work.svg)

## Start mps control daemon
```bash
# start MPS on gpu #3
sudo env GPU_ID=3 bash start_mps.bash
```

## Run your CUDA application
```bash
export GPU_ID=3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$GPU_ID
# here is a testing script I wrote
python cuda_app.py --gpu 0
```

Note that via MPS, the CUDA application would see only one GPU. Try to run `python cuda_app.py --gpu 3` in the above snippet and see what happen.


## Stop mps control daemon
```bash
sudo env GPU_ID=3 bash stop_mps.bash
```

## Pass command to mps control daemon
```bash
env GPU_ID=3 bash cmd_mps.bash <command>
```
see `man 1 nvidia-cuda-mps-control` for a list of available commands
