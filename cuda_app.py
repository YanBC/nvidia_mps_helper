import argparse
import pycuda.driver as cuda

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, required=True)
args = p.parse_args()

gpu_id = args.gpu
cuda.init()
device = cuda.Device(gpu_id)
context = device.make_context()
print(context.get_api_version())
context.pop()
