"""Run a handful of traced dispatches and print the SM-level timeline of rank 0. Debug / sanity helper."""
import argparse, os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_layout as TL


def worker(local_rank, n, args):
    import deep_ep
    from deep_ep.utils.envs import init_dist
    rank, num_ranks, group = init_dist(local_rank, n, seed=0)
    buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=args.num_max_tokens, hidden=7168, allow_hybrid_mode=False,
                              allow_multiple_reduction=True, explicitly_destroy=True, num_gpu_timeout_secs=20, num_cpu_timeout_secs=20)
    nle = 256 // num_ranks; T = args.num_tokens
    x = torch.randn(T, 7168, dtype=torch.bfloat16, device='cuda')
    scores = torch.rand(T, 256, device='cuda')
    w, idx = torch.topk(scores, 8, dim=-1); idx = idx.to(deep_ep.topk_idx_t); w = w.float()
    stats, trace = TL.alloc_trace(nle, T, 'cuda')
    if not args.trace: trace[TL.HDR] = 0
    for it in range(args.iters):
        TL.reset_trace(trace, T) if args.trace else None
        t0 = time.time()
        buffer.dispatch(x, topk_idx=idx, topk_weights=w, cumulative_local_expert_recv_stats=stats, num_experts=256,
                        num_max_tokens_per_rank=args.num_max_tokens, expert_alignment=128, num_sms=args.num_sms,
                        async_with_compute_stream=False, do_cpu_sync=True, do_expand=True, do_zero_padding=False, defer_epilogue=False)
        torch.cuda.synchronize()
        if rank == 0: print(f'iter {it}: {(time.time()-t0)*1e3:.2f} ms wall', flush=True)
    dist.barrier()
    if rank == 0 and args.trace:
        tr = trace.cpu(); base = int(tr[TL.SM]);
        for e, name in TL.SM_EVENTS.items():
            vals = [int(tr[TL.SM + s * 32 + e]) for s in range(args.num_sms)]
            print(f'{name:22s} ' + ' '.join(f'{(v-base)/1e3:8.1f}' if v else '       -' for v in vals[:8]) + '  (us rel. to SM0 entry, first 8 SMs)')
        tok = tr[TL.TOK:TL.TOK + TL.TOK_STRIDE * T].view(T, TL.TOK_STRIDE)
        print('token 0 record:', tok[0].tolist()[:12])
        print('token 0 lane issue:', [(v - base) / 1e3 if v else None for v in tok[0, 8:16].tolist()])
        print('token 0 lane done :', [(v - base) / 1e3 if v else None for v in tok[0, 16:24].tolist()])
        print('token 0 lane dst  :', tok[0, 24:32].tolist())
    buffer.destroy(); dist.destroy_process_group()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--num-tokens', type=int, default=512); ap.add_argument('--num-max-tokens', type=int, default=4096)
    ap.add_argument('--num-sms', type=int, default=4); ap.add_argument('--iters', type=int, default=3); ap.add_argument('--trace', type=int, default=1)
    ap.add_argument('--num-processes', type=int, default=4)
    a = ap.parse_args(); mp.spawn(worker, args=(a.num_processes, a), nprocs=a.num_processes)
