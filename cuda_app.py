import argparse
import pycuda.driver as cuda

cuda.init()
print(f"num of available GPUs: {cuda.Device.count()}")

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int)
args = p.parse_args()

if args.gpu is not None:
    gpu_id = args.gpu
    device = cuda.Device(gpu_id)
    context = device.make_context()
    print(context.get_api_version())
    context.pop()
