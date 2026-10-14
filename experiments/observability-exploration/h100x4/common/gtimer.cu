// Tiny helper: read %globaltimer on the GPU so that host timestamps can be mapped onto the GPU clock.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void gt_kernel(long long* out) { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); *out = (long long)t; }
__global__ void spin_kernel(long long ns) { unsigned long long t0, t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0)); do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); } while ((long long)(t - t0) < ns); }
int64_t read_globaltimer(torch::Tensor pinned) { gt_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<long long*>(pinned.data_ptr<int64_t>())); cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()); return pinned.item<int64_t>(); }
void spin(int64_t ns) { spin_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(ns); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("read_globaltimer", &read_globaltimer); m.def("spin", &spin); }
