"""E1 driver. Single process, all visible GPUs. Produces results/e1_results.json.

Usage: CUDA_VISIBLE_DEVICES=4,5,6,7 python e1_run.py [--quick]
"""
import argparse, json, os, sys, time, statistics
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
os.makedirs(os.path.join(HERE, 'build'), exist_ok=True)

ext = load(name='e1_bulk', sources=[os.path.join(HERE, 'e1_bulk.cu')], build_directory=os.path.join(HERE, 'build'),
           extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-std=c++20'], verbose=False)

MODES = {0: 'wait_group', 1: 'wait_group.read', 2: 'wait_group+fence.sys', 3: 'wait_group.read+fence.sys', 4: 'st.global x32 lanes+fence.sys', 5: 'wait_group+fence.gpu'}


def pct(xs, p):
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))); return xs[k]


def summarize(cyc, ns, calib, warm=200):
    # calibrate cycles -> ns per slot using the whole-run (clock64, globaltimer) pair
    cyc = cyc.cpu().view(calib.numel() // 4, -1); ns = ns.cpu().view_as(cyc); calib = calib.cpu().view(-1, 4)
    out = []
    for s in range(cyc.shape[0]):
        c0, n0, c1, n1 = calib[s].tolist(); ghz = (c1 - c0) / max(1, (n1 - n0))  # cycles per ns
        lat_ns = (cyc[s, warm:].double() / ghz).tolist()
        out.append(dict(ghz=ghz, p50=pct(lat_ns, 50), p10=pct(lat_ns, 10), p90=pct(lat_ns, 90), p99=pct(lat_ns, 99), mean=statistics.fmean(lat_ns), n=len(lat_ns), elapsed_ns=(n1 - n0)))
    agg = dict(p50=statistics.median([o['p50'] for o in out]), p99=max(o['p99'] for o in out), mean=statistics.fmean([o['mean'] for o in out]), ghz=statistics.fmean([o['ghz'] for o in out]),
               elapsed_ns=max(o['elapsed_ns'] for o in out), per_slot=[{k: v for k, v in o.items() if k != 'n'} for o in out])
    return agg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); args = ap.parse_args()
    nd = torch.cuda.device_count(); assert nd >= 2
    devs = list(range(nd))
    for a in devs:
        for b in devs:
            if a != b: ext.enable_peer(a, b)
    props = torch.cuda.get_device_properties(0)
    results = dict(meta=dict(num_devices=nd, device=props.name, cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), time=time.strftime('%Y-%m-%d %H:%M:%S')), latency=[], issue=[], pingpong=[], visibility=[])
    iters = 400 if args.quick else 2000
    sizes = [16, 256, 4096, 14464, 65536] if args.quick else [16, 64, 256, 1024, 4096, 14464, 32768, 65536]
    region_bytes = 256 << 20  # 256 MB destination region per device
    bufs = {d: torch.empty(region_bytes, dtype=torch.int8, device=f'cuda:{d}') for d in devs}

    # ---- A. single-warp latency: destination self and each peer, every mode, every size ----
    src = 0
    for dst in devs:
        for mode in ([0, 1, 2, 4] if args.quick else [0, 1, 2, 3, 4, 5]):
            for b in sizes:
                if mode == 4 and b > 65536: continue
                cyc, ns, calib = ext.bulk_latency(src, bufs[dst], b, iters, mode, 1, 1)
                s = summarize(cyc, ns, calib)
                results['latency'].append(dict(src=src, dst=dst, remote=dst != src, mode=mode, mode_name=MODES[mode], bytes=b, blocks=1, warps=1, **{k: s[k] for k in ('p50', 'p99', 'mean', 'ghz')}))
                print(f'lat src{src}->dst{dst} {MODES[mode]:28s} {b:6d} B  p50 {s["p50"]:8.0f} ns  p99 {s["p99"]:8.0f} ns', flush=True)

    # ---- B. concurrency: 14,464 B stores, remote destination, 1..15 warps on 1 SM and 15 warps on 1..16 SMs ----
    dst = 1
    for warps in [1, 4, 8, 15]:
        cyc, ns, calib = ext.bulk_latency(src, bufs[dst], 14464, iters, 0, 1, warps)
        s = summarize(cyc, ns, calib)
        delivered = warps * iters * 14464 / s['elapsed_ns']
        results['latency'].append(dict(src=src, dst=dst, remote=True, mode=0, mode_name=MODES[0], bytes=14464, blocks=1, warps=warps, delivered_gbs=delivered, **{k: s[k] for k in ('p50', 'p99', 'mean', 'ghz', 'elapsed_ns')}))
        print(f'conc 1 SM x {warps:2d} warps  p50 {s["p50"]:8.0f} ns  p99 {s["p99"]:8.0f} ns  issue-rate {warps*14464/ s["mean"]:.1f} GB/s  delivered(incl. drain) {delivered:.1f} GB/s', flush=True)
    for blocks in [1, 2, 4, 8, 16]:
        cyc, ns, calib = ext.bulk_latency(src, bufs[dst], 14464, iters, 0, blocks, 15)
        s = summarize(cyc, ns, calib)
        delivered = blocks * 15 * iters * 14464 / s['elapsed_ns']
        results['latency'].append(dict(src=src, dst=dst, remote=True, mode=0, mode_name=MODES[0], bytes=14464, blocks=blocks, warps=15, delivered_gbs=delivered, **{k: s[k] for k in ('p50', 'p99', 'mean', 'ghz', 'elapsed_ns')}))
        print(f'conc {blocks:2d} SMs x 15 warps  p50 {s["p50"]:8.0f} ns  p99 {s["p99"]:8.0f} ns  issue-rate {blocks*15*14464/ s["mean"]:.1f} GB/s  delivered(incl. drain) {delivered:.1f} GB/s', flush=True)
    # concurrency with a system fence per store (true remote completion) and with a local destination
    for (dst_, mode_) in [(1, 2), (0, 0)]:
        for blocks in [1, 4, 16]:
            cyc, ns, calib = ext.bulk_latency(src, bufs[dst_], 14464, iters, mode_, blocks, 15)
            s = summarize(cyc, ns, calib)
            delivered = blocks * 15 * iters * 14464 / s['elapsed_ns']
            results['latency'].append(dict(src=src, dst=dst_, remote=dst_ != src, mode=mode_, mode_name=MODES[mode_], bytes=14464, blocks=blocks, warps=15, delivered_gbs=delivered, **{k: s[k] for k in ('p50', 'p99', 'mean', 'ghz', 'elapsed_ns')}))
            print(f'conc {blocks:2d} SMs x 15 warps dst{dst_} {MODES[mode_]:22s} p50 {s["p50"]:8.0f} ns p99 {s["p99"]:8.0f} ns  delivered {delivered:.1f} GB/s', flush=True)

    # ---- C. issue without waiting: credit limit ----
    for dst in [0, 1]:
        for b in [1024, 14464, 65536]:
            for rep in range(3):
                issue, done = ext.issue_rate(src, bufs[dst], b, 512)
                ghz = results['latency'][0]['ghz']
                issue_ns = (issue.cpu().double() / ghz).tolist(); done_ns = done.item() / ghz
                # knee: first index where the per-issue gap exceeds 5x the median of the first 8 gaps
                gaps = [issue_ns[i] - (issue_ns[i - 1] if i else 0) for i in range(len(issue_ns))]
                base = statistics.median(gaps[1:9]); knee = next((i for i, g in enumerate(gaps) if i > 8 and g > 5 * base), None)
                results['issue'].append(dict(src=src, dst=dst, remote=dst != src, bytes=b, rep=rep, issue_ns=issue_ns, done_ns=done_ns, base_gap_ns=base, knee_index=knee))
                print(f'issue src{src}->dst{dst} {b:6d} B: base gap {base:6.0f} ns, knee at {knee}, all 512 done at {done_ns/1e3:8.1f} us', flush=True)

    # ---- D. ping-pong RTT and clock offsets between every pair ----
    flags = {d: torch.zeros(4, dtype=torch.int32, device=f'cuda:{d}') for d in devs}
    pp_iters = 2000 if args.quick else 10000
    for a in devs:
        for b in devs:
            if a == b: continue
            ra_ns, ra_cyc, rb_ns, rb_cyc = ext.pingpong(a, flags[a], b, flags[b], pp_iters)
            ra = ra_ns.cpu().view(-1, 2)[200:]; rb = rb_ns.cpu().view(-1, 2)[200:]
            rtt = (ra[:, 1] - ra[:, 0]).tolist()
            # PTP-style offset of B's clock relative to A's: offset = ((tB_recv - tA_send) - (tA_recv - tB_send)) / 2
            off = (((rb[:, 0] - ra[:, 0]) - (ra[:, 1] - rb[:, 1])) / 2).double()
            results['pingpong'].append(dict(a=a, b=b, rtt_p50=pct(rtt, 50), rtt_p10=pct(rtt, 10), rtt_p99=pct(rtt, 99), offset_b_minus_a_median=off.median().item(), offset_std=off.std().item(), n=len(rtt)))
            print(f'pingpong {a}<->{b}: RTT p50 {pct(rtt,50):6.0f} ns p99 {pct(rtt,99):6.0f} ns | offset(b-a) {off.median().item():10.0f} ns +- {off.std().item():6.0f}', flush=True)

    # ---- E. visibility: does the payload already hold the new pattern when the receiver sees the flag, per wait mode ----
    for mode in [0, 1, 2]:
        for b in [4096, 14464, 65536]:
            payload = bufs[1][:b]
            stale = ext.visibility(0, 1, payload, flags[1], b, 2000, mode).cpu()[1:]
            chunks = b // 16
            results['visibility'].append(dict(mode=mode, mode_name=MODES[mode], bytes=b, rounds=int(stale.numel()), rounds_with_stale=int((stale > 0).sum()), max_stale_chunks=int(stale.max()), chunks=chunks))
            print(f'visibility {MODES[mode]:28s} {b:6d} B: rounds with stale data {(stale>0).sum().item()}/{stale.numel()}  max stale chunks {stale.max().item()}/{chunks}', flush=True)

    with open(os.path.join(HERE, 'results', 'e1_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', os.path.join(HERE, 'results', 'e1_results.json'))


if __name__ == '__main__':
    main()
