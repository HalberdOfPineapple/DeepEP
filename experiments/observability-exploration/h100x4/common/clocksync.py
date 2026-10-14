"""In-process cross-GPU clock calibration for multi-process runs.

Each rank owns one GPU. Flags live in torch symmetric memory so that every rank can write into every other rank's
flag word with a plain store. For every ordered pair (a, b) the two ranks run the ping-pong kernel from E1's extension
(initiator a, responder b) while the others wait; the PTP-style offset clock_b - clock_a is computed from both sides'
globaltimer stamps. Returns {(a, b): offset_ns} plus RTT medians. Cost: ~1 ms per pair.
"""
import os, sys, statistics
import torch, torch.distributed as dist

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__)); e1 = os.path.join(os.path.dirname(here), 'E1-bulk-store-completion')
        _ext = load(name='e1_bulk', sources=[os.path.join(e1, 'e1_bulk.cu')], build_directory=os.path.join(e1, 'build'),
                    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-std=c++20'], verbose=False)
    return _ext


class ClockSync:
    def __init__(self, group, rank, world):
        import torch.distributed._symmetric_memory as symm_mem
        self.ext = _load(); self.rank, self.world, self.group = rank, world, group
        self.flags = symm_mem.empty(64, dtype=torch.int32, device='cuda'); self.flags.zero_()
        self.hdl = symm_mem.rendezvous(self.flags, group)
        self.peer_ptrs = [self.hdl.get_buffer(p, (64,), torch.int32).data_ptr() for p in range(world)]
        torch.cuda.synchronize(); dist.barrier(group)

    def measure(self, iters=400):
        """Returns dict with 'offset' {(a,b): ns (clock_b - clock_a)}, 'rtt' {(a,b): ns}, 'host_time'."""
        import time
        local = {}
        for a in range(self.world):
            for b in range(self.world):
                if a == b: continue
                # slot for this pair: word index a (initiator writes peer's word a? keep one word per pair: idx = a*world+b)
                idx = a * self.world + b
                self.flags.zero_(); torch.cuda.synchronize(); dist.barrier(self.group)
                if self.rank == b:
                    rec_ns, _ = self.ext.pingpong_side(self.flags.data_ptr() + 4 * idx, self.peer_ptrs[a] + 4 * idx, iters, 0)
                    local[(a, b, 'B')] = rec_ns.cpu().view(-1, 2)[50:].clone()
                elif self.rank == a:
                    rec_ns, _ = self.ext.pingpong_side(self.flags.data_ptr() + 4 * idx, self.peer_ptrs[b] + 4 * idx, iters, 1)
                    local[(a, b, 'A')] = rec_ns.cpu().view(-1, 2)[50:].clone()
                dist.barrier(self.group)
        gathered = [None] * self.world
        dist.all_gather_object(gathered, local, group=self.group)
        allrec = {}
        for g in gathered: allrec.update(g)
        offset, rtt = {}, {}
        for a in range(self.world):
            for b in range(self.world):
                if a == b: continue
                ra, rb = allrec[(a, b, 'A')], allrec[(a, b, 'B')]
                # integer arithmetic only: deep_ep sets the default dtype to bfloat16, so '/ 2' would quantise to 2^23 ns
                off = ((rb[:, 0] - ra[:, 0]) - (ra[:, 1] - rb[:, 1])) // 2
                offset[(a, b)] = int(off.median().item()); rtt[(a, b)] = int((ra[:, 1] - ra[:, 0]).median().item())
        return dict(offset=offset, rtt=rtt, host_time=time.time())
