"""Run DeepEP dispatch on the visible GPUs with optional per-lane tracing and save results.

Example:
  EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=common/pkg python common/run_dispatch.py \
      --num-sms 16 --num-tokens 4096 --ratio 1 --repeats 5 --out E0-window-vs-links/results/sms16_tok4096.pt

Each output .pt holds: args, per-rank kernel times (kineto, untraced), per-rank traced-run data (trace tensor,
topk_idx, dispatch kernel time of the traced run), and the actual destination-rank counts.
"""
import argparse, os, sys, time, json
import torch, torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_layout as TL


def worker(local_rank, num_local_ranks, args):
    import deep_ep
    from deep_ep.utils.envs import init_dist
    from deep_ep.utils.testing import bench_kineto
    from deep_ep.utils.gate import get_unbalanced_scores
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    torch.manual_seed(args.seed * 1000 + rank)
    buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=args.num_max_tokens, hidden=args.hidden,
                              allow_hybrid_mode=False, allow_multiple_reduction=True, explicitly_destroy=True,
                              num_gpu_timeout_secs=60, num_cpu_timeout_secs=60)
    num_local_experts = args.num_experts // num_ranks
    T = args.num_tokens
    x = torch.randn(T, args.hidden, dtype=torch.bfloat16, device='cuda')
    scores = get_unbalanced_scores(T, args.num_experts, num_ranks, args.topk, args.ratio, bool(args.precise))
    topk_weights, topk_idx = torch.topk(scores, args.topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = topk_weights.float()
    # destination-rank statistics for this sender
    dst_rank = topk_idx // num_local_experts
    uniq_per_token = torch.stack([(dst_rank == r).any(dim=1) for r in range(num_ranks)], dim=1)
    dst_counts = uniq_per_token.sum(dim=0).cpu()  # tokens sent to each rank
    k_mean = uniq_per_token.sum(dim=1).float().mean().item()

    stats_off, trace_off = TL.alloc_trace(num_local_experts, T, 'cuda'); trace_off[TL.HDR] = 0  # tracing disabled (no magic)
    stats_on, trace_on = TL.alloc_trace(num_local_experts, T, 'cuda')

    def run(stats):
        stats[:num_local_experts].zero_()
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights, cumulative_local_expert_recv_stats=stats,
            num_experts=args.num_experts, num_max_tokens_per_rank=args.num_max_tokens, expert_alignment=128,
            num_sms=args.num_sms, async_with_compute_stream=False, do_cpu_sync=True, do_expand=True,
            do_zero_padding=False, defer_epilogue=False)
        return handle

    # warm-up (JIT compile + first-touch)
    for _ in range(3):
        run(stats_off)
    torch.cuda.synchronize(); dist.barrier()

    # untraced kernel time via kineto (same method as DeepEP's test)
    if not args.no_kineto:
        t_off = bench_kineto(lambda: run(stats_off), 'dispatch_impl<', num_tests=args.num_tests, suppress_kineto_output=True, flush_l2=bool(args.flush_l2), barrier_comm_profiling=True)
        t_on = bench_kineto(lambda: run(stats_on), 'dispatch_impl<', num_tests=args.num_tests, suppress_kineto_output=True, flush_l2=bool(args.flush_l2), barrier_comm_profiling=True)
        t_off = float(t_off[0] if isinstance(t_off, (tuple, list)) else t_off); t_on = float(t_on[0] if isinstance(t_on, (tuple, list)) else t_on)
    else:
        t_off = t_on = float('nan')

    # ---- optional machinery for E5-E9 ----
    bg_stream = torch.cuda.Stream(); bg_on = args.bg != 'none' and (args.bg_ranks == 'all' or rank in [int(v) for v in args.bg_ranks.split(',')])
    if bg_on:
        if args.bg == 'gemm':
            # short kernels (~30 us) so that the cooperative dispatch launch can still find free SMs
            A = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda'); B = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda'); C = torch.empty_like(A)
            def bg_kernel(): torch.matmul(A, B, out=C)
        else:
            # 32 MB copies (~20 us each at HBM rate)
            S = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device='cuda'); D = torch.empty_like(S)
            def bg_kernel(): D.copy_(S)
        # calibrate: time of one bg kernel alone
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); [bg_kernel() for _ in range(10)]; e1.record(); torch.cuda.synchronize(); bg_ms_each = e0.elapsed_time(e1) / 10
    else:
        bg_ms_each = 0.0
    hc = None
    if args.host_ts or (args.skew_mode == 'gpu' and args.skew_rank >= 0):
        from hostclock import HostClock
        hc = HostClock()
    ov_handle = None; ov_group = None; ov_buf = None
    if args.overlap == 'combine':
        ov_buf = deep_ep.EPBuffer(group, num_max_tokens_per_rank=args.num_max_tokens, hidden=args.hidden, allow_hybrid_mode=False,
                                  allow_multiple_reduction=True, explicitly_destroy=True, num_gpu_timeout_secs=60, num_cpu_timeout_secs=60)
        ov_stats = torch.zeros(num_local_experts, dtype=torch.int32, device='cuda')
        ov_recv_x, _, ov_recv_w, ov_handle, _ = ov_buf.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, cumulative_local_expert_recv_stats=ov_stats,
            num_experts=args.num_experts, num_max_tokens_per_rank=args.num_max_tokens, expert_alignment=128, num_sms=args.num_sms,
            async_with_compute_stream=False, do_cpu_sync=True, do_expand=True, do_zero_padding=False, defer_epilogue=False)
        ov_x = ov_recv_x.to(torch.bfloat16) if ov_recv_x.dtype != torch.bfloat16 else ov_recv_x
        for _ in range(3): ov_buf.combine(x=ov_x, handle=ov_handle, topk_weights=ov_recv_w, num_sms=args.num_sms)
        torch.cuda.synchronize()
        # overlapped combine runs on the side stream (its own comm stream synchronises with the side stream)
        def overlap_launch():
            with torch.cuda.stream(bg_stream):
                ov_buf.combine(x=ov_x, handle=ov_handle, topk_weights=ov_recv_w, num_sms=args.num_sms)
        oe0 = torch.cuda.Event(enable_timing=True); oe1 = torch.cuda.Event(enable_timing=True)
        oe0.record(); ov_buf.combine(x=ov_x, handle=ov_handle, topk_weights=ov_recv_w, num_sms=args.num_sms); oe1.record(); torch.cuda.synchronize(); ov_alone_ms = oe0.elapsed_time(oe1)
    elif args.overlap == 'nccl':
        ov_group = dist.new_group(backend='nccl')
        a2a_in = torch.randn(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device='cuda'); a2a_out = torch.empty_like(a2a_in)
        for _ in range(3): dist.all_to_all_single(a2a_out, a2a_in, group=ov_group)
        torch.cuda.synchronize()
        def overlap_launch():
            with torch.cuda.stream(bg_stream):
                for _ in range(args.bg_iters): dist.all_to_all_single(a2a_out, a2a_in, group=ov_group, async_op=True)
        oe0 = torch.cuda.Event(enable_timing=True); oe1 = torch.cuda.Event(enable_timing=True)
        oe0.record(); [dist.all_to_all_single(a2a_out, a2a_in, group=ov_group) for _ in range(5)]; oe1.record(); torch.cuda.synchronize(); ov_alone_ms = oe0.elapsed_time(oe1) / 5
    else:
        overlap_launch = None; ov_alone_ms = 0.0
    sampler = None
    if args.nvml and rank == 0:
        from nvml_links import LinkSampler
        vis = [int(v) for v in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')]
        sampler = LinkSampler(vis, period_s=0.001)

    # traced repeats, each bracketed by a cross-GPU clock calibration (offsets drift between runs)
    from clocksync import ClockSync
    cs = ClockSync(group, rank, num_ranks)
    traces, clocks, host_ts, ov_ms, nvml_samples = [], [], [], [], []
    ev_a = torch.cuda.Event(enable_timing=True); ev_b = torch.cuda.Event(enable_timing=True)
    for r in range(args.repeats):
        c0 = cs.measure()
        TL.reset_trace(trace_on, T); torch.cuda.synchronize()
        hts = {}
        if hc is not None: hts['map0'] = hc.sample()
        if sampler is not None: sampler.start()   # the NVML read takes tens of ms: do it before the barrier so only this rank pays
        dist.barrier()
        if bg_on:
            with torch.cuda.stream(bg_stream):
                for _ in range(args.bg_iters): bg_kernel()
        if overlap_launch is not None:
            ev_a.record(bg_stream); overlap_launch(); ev_b.record(bg_stream)
        if args.skew_rank == rank and args.skew_us > 0:
            if args.skew_mode == 'host':
                t_end = time.perf_counter_ns() + int(args.skew_us * 1000)
                while time.perf_counter_ns() < t_end: pass
            else:
                hc.spin_us(args.skew_us)
        stats_on[:num_local_experts].zero_()
        hts['call'] = time.perf_counter_ns()
        handle = buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, cumulative_local_expert_recv_stats=stats_on,
            num_experts=args.num_experts, num_max_tokens_per_rank=args.num_max_tokens, expert_alignment=128,
            num_sms=args.num_sms, async_with_compute_stream=False, do_cpu_sync=bool(args.cpu_sync), do_expand=True,
            do_zero_padding=False, defer_epilogue=False)[3]
        hts['ret'] = time.perf_counter_ns()
        torch.cuda.synchronize(); hts['sync'] = time.perf_counter_ns()
        if sampler is not None: nvml_samples.append(sampler.stop())
        if hc is not None: hts['map1'] = hc.sample()
        if overlap_launch is not None: ov_ms.append(ev_a.elapsed_time(ev_b))
        host_ts.append(hts)
        traces.append(trace_on.detach().cpu().clone())
        c1 = cs.measure()
        clocks.append(dict(before=c0, after=c1))
    dist.barrier()
    if ov_buf is not None: ov_buf.destroy()
    num_recv = int(stats_on[:num_local_experts].sum().item())

    auto_sms = buffer.get_theoretical_num_sms(args.num_experts, args.topk)
    payload = dict(rank=rank, num_ranks=num_ranks, auto_sms=auto_sms, t_untraced_s=t_off, t_traced_s=t_on, k_mean=k_mean,
                   dst_counts=dst_counts, num_recv_token_expert_pairs=num_recv, topk_idx=topk_idx.cpu(), traces=traces, clocks=clocks,
                   num_dispatch_warps=None, host_ts=host_ts, bg=dict(on=bg_on, kind=args.bg, ms_each=bg_ms_each, iters=args.bg_iters),
                   overlap=dict(kind=args.overlap, alone_ms=ov_alone_ms, overlapped_ms=ov_ms), nvml=nvml_samples)
    gathered = [None] * num_ranks
    dist.all_gather_object(gathered, payload)
    if rank == 0:
        out = dict(args=vars(args), ranks=gathered, time=time.strftime('%Y-%m-%d %H:%M:%S'))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(out, args.out)
        bytes_per_token = (args.hidden * 2 + 127) // 128 * 128  # approximate token row incl. metadata, see TokenLayout
        for g in gathered:
            remote_tokens = int(g['dst_counts'].sum()) - int(g['dst_counts'][g['rank']])
            print(f"rank {g['rank']}: dispatch {g['t_untraced_s']*1e6:8.1f} us untraced, {g['t_traced_s']*1e6:8.1f} us traced "
                  f"({(g['t_traced_s']/g['t_untraced_s']-1)*100:+.1f}%), E[k]={g['k_mean']:.2f}, dst_counts={g['dst_counts'].tolist()}, "
                  f"remote GB/s ~ {remote_tokens*14464/g['t_untraced_s']/1e9:.0f}", flush=True)
        print(f"auto num_sms would be {gathered[0]['auto_sms']}; saved", args.out, flush=True)
    buffer.destroy(); dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-processes', type=int, default=4)
    ap.add_argument('--num-sms', type=int, default=16)
    ap.add_argument('--num-tokens', type=int, default=4096)
    ap.add_argument('--num-max-tokens', type=int, default=4096)
    ap.add_argument('--hidden', type=int, default=7168)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--num-experts', type=int, default=256)
    ap.add_argument('--ratio', type=float, default=1.0)
    ap.add_argument('--precise', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--num-tests', type=int, default=20)
    ap.add_argument('--flush-l2', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, required=True)
    # --- extensions for E5-E9 ---
    ap.add_argument('--bg', type=str, default='none', help='background load on a side stream during traced dispatches: none | gemm | copy')
    ap.add_argument('--bg-ranks', type=str, default='all', help='all, or comma list of ranks that run the background load')
    ap.add_argument('--bg-iters', type=int, default=40, help='background kernels queued per traced dispatch')
    ap.add_argument('--skew-rank', type=int, default=-1, help='rank that is delayed before each traced dispatch')
    ap.add_argument('--skew-us', type=float, default=0.0)
    ap.add_argument('--skew-mode', type=str, default='host', help='host: sleep on the host before the launch; gpu: spin kernel on the stream before the launch')
    ap.add_argument('--cpu-sync', type=int, default=1, help='do_cpu_sync argument of dispatch')
    ap.add_argument('--host-ts', type=int, default=0, help='record host timestamps around each traced dispatch and map them onto the GPU clock')
    ap.add_argument('--overlap', type=str, default='none', help='none | combine (a second buffer runs combine concurrently) | nccl (a NCCL all_to_all runs concurrently)')
    ap.add_argument('--nvml', type=int, default=0, help='sample per-link NVLink counters on the visible GPUs during traced repeats (rank 0)')
    ap.add_argument('--no-kineto', type=int, default=0, help='skip the Kineto timing runs (E9 decode uses in-trace spans)')
    args = ap.parse_args()
    mp.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == '__main__':
    main()
