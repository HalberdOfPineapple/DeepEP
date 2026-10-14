"""Background sampler of per-link NVLink data counters (KiB) for a set of GPU indices, at a fixed period."""
import threading, time
class LinkSampler:
    def __init__(self, gpus, period_s=0.001, links=18):
        import pynvml as n
        self.n = n; n.nvmlInit(); self.h = {g: n.nvmlDeviceGetHandleByIndex(g) for g in gpus}; self.gpus = gpus; self.period = period_s; self.links = links
        self.samples = []; self._stop = threading.Event(); self.t = None
    def read(self):
        n = self.n; out = {}
        for g, h in self.h.items():
            tx = n.nvmlDeviceGetFieldValues(h, [(n.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, l) for l in range(self.links)])
            rx = n.nvmlDeviceGetFieldValues(h, [(n.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, l) for l in range(self.links)])
            out[g] = ([v.value.ullVal for v in tx], [v.value.ullVal for v in rx])
        return out
    # one NVML read of 4 GPUs x 36 fields takes several ms, so the sampler records a reading at start and one at stop
    def start(self): self.samples = [(time.perf_counter_ns(), self.read())]
    def stop(self): self.samples.append((time.perf_counter_ns(), self.read())); return self.samples
