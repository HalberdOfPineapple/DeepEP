import torch, torch.multiprocessing as mp
import deep_ep
from deep_ep import EPBuffer
from deep_ep.utils.envs import init_dist

HIDDEN, FFN, NUM_EXPERTS, TOPK = 7168, 2048, 256, 8
MAX_TOKENS, ALIGN = 4096, 128


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, buffer: EPBuffer, num_sms):
        # ② dispatch: route each token to the ranks owning its top-k experts
        event = buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_experts=NUM_EXPERTS, num_max_tokens_per_rank=MAX_TOKENS,
            expert_alignment=ALIGN, num_sms=num_sms,
            async_with_compute_stream=True, do_cpu_sync=True,
            do_expand=True, do_zero_padding=True, defer_epilogue=True
        )
        recv_x, _, recv_w, handle = event.current_stream_wait()   # runs the deferred copy epilogue
        ctx.buffer, ctx.handle, ctx.num_sms = buffer, handle, num_sms
        return recv_x, recv_w, handle

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_w, _):
        # ⑤ backward of dispatch = combine of the gradients
        gx, gw, event = ctx.buffer.combine(grad_recv_x, handle=ctx.handle, topk_weights=grad_recv_w,
                                           num_sms=ctx.num_sms, async_with_compute_stream=True)
        event.current_stream_wait()
        return gx, None, gw, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, shared_out, handle, buffer, num_sms):
        # ④ combine: send expert outputs home and sum them (+ shared expert as bias)
        event = buffer.combine(y, handle=handle, bias=shared_out, num_sms=num_sms,
                               async_with_compute_stream=True, defer_epilogue=True)
        out, _ = event.current_stream_wait()
        ctx.buffer, ctx.handle, ctx.num_sms = buffer, handle, num_sms
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # ⑤ backward of combine = cached dispatch (same routes, no new counting)
        grad_y, _, _, _, event = ctx.buffer.dispatch(grad_out, handle=ctx.handle, num_sms=ctx.num_sms,
                                                     async_with_compute_stream=True,
                                                     do_expand=True, do_zero_padding=True)
        event.current_stream_wait()
        return grad_y, grad_out, None, None, None   # d(out)/d(bias) = identity


class MoELayer(torch.nn.Module):
    def __init__(self, buffer, rank, num_ranks):
        super().__init__()
        self.buffer, self.num_local = buffer, NUM_EXPERTS // num_ranks          # 32 per GPU
        self.num_sms = buffer.get_theoretical_num_sms(NUM_EXPERTS, TOPK)       # 16 on 8×H100
        self.gate = torch.nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.shared = torch.nn.Sequential(torch.nn.Linear(HIDDEN, FFN), torch.nn.SiLU(), torch.nn.Linear(FFN, HIDDEN))
        self.w_up = torch.nn.Parameter(torch.randn(self.num_local, HIDDEN, FFN) * 0.02)
        self.w_down = torch.nn.Parameter(torch.randn(self.num_local, FFN, HIDDEN) * 0.02)

    def forward(self, x):                                                      # x: [T, 7168] bf16, T ≤ 4096
        # ① gating stays local
        topk_w, topk_idx = self.gate(x).float().softmax(-1).topk(TOPK, dim=-1)
        recv_x, recv_w, handle = Dispatch.apply(x, topk_idx.to(deep_ep.topk_idx_t), topk_w,
                                                self.buffer, self.num_sms)
        shared_out = self.shared(x)

        # ③ local experts: rows are grouped per expert, each segment padded to ALIGN
        outs, start = [], 0
        for e, n in enumerate(handle.num_recv_tokens_per_expert_list):
            h = recv_x[start:start + n]
            outs.append(torch.nn.functional.silu(h @ self.w_up[e]) @ self.w_down[e])
            start += n
        y = (torch.cat(outs) * recv_w.unsqueeze(1)).to(x.dtype)               # router weights applied HERE
        return Combine.apply(y, shared_out, handle, self.buffer, self.num_sms)


def worker(local_rank, num_local_ranks):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)          # NCCL group of 8
    buffer = EPBuffer(group, num_max_tokens_per_rank=MAX_TOKENS, hidden=HIDDEN,
                      num_topk=TOPK, explicitly_destroy=True)                 # ⓪ start-up
    layer = MoELayer(buffer, rank, num_ranks)
    x = torch.randn(MAX_TOKENS - rank, HIDDEN, requires_grad=True)          # ranks may differ in T
    layer(x).sum().backward()
    buffer.destroy()                                                         # barrier, free window
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(8,), nprocs=8)