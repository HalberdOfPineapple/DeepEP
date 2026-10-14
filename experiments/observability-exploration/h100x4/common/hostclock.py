"""Map host perf_counter_ns onto the GPU %globaltimer: launch a 1-thread kernel that reads the timer, bracket it with host stamps,
keep the sample with the tightest bracket. Also exposes a GPU-side spin (busy wait on the stream) for skew injection."""
import os, time, torch

_ext = None
def load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load as _load
        here = os.path.dirname(os.path.abspath(__file__))
        _ext = _load(name='gtimer', sources=[os.path.join(here, 'gtimer.cu')], build_directory=os.path.join(here, 'build_gtimer'),
                     extra_cuda_cflags=['-O3'], extra_include_paths=[], verbose=False)
    return _ext

class HostClock:
    def __init__(self):
        self.ext = load(); self.pinned = torch.zeros(1, dtype=torch.int64, device='cpu').pin_memory()
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_gtimer'), exist_ok=True)
    def sample(self, n=50):
        """Returns (host_ns, gpu_ns, half_bracket_ns): gpu_ns ≈ host_ns + offset at the instant of the kernel."""
        best = None
        for _ in range(n):
            h0 = time.perf_counter_ns(); g = self.ext.read_globaltimer(self.pinned); h1 = time.perf_counter_ns()
            if best is None or (h1 - h0) < best[2] * 2: best = ((h0 + h1) // 2, g, (h1 - h0) // 2)
        return best
    def spin_us(self, us): self.ext.spin(int(us * 1000))
