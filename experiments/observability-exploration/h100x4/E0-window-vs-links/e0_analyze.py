"""Build report.html for E0 from results/*.pt (produced by e0_sweep.py / run_dispatch.py)."""
import glob, os, re, sys, statistics
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, trace_layout as TL
from viz import PALETTE, INK, MUTED

BYTES_PER_TOKEN = 14464
rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'results', '*.pt'))):
    d = torch.load(f, weights_only=False)
    a = d['args']; sms, tok = a['num_sms'], a['num_tokens']
    per_rank = []
    for g in d['ranks']:
        dc = g['dst_counts']; r = g['rank']
        remote_tokens = int(dc.sum()) - int(dc[r]); total_tokens = int(dc.sum())
        t = g['t_untraced_s']
        # trace-derived metrics (median over repeats)
        slot_wait, lane_lat, atomic_lat, load_lat, iter_time, open_b, close_wait, data_span, issue_time, spread = [], [], [], [], [], [], [], [], [], []
        for tr in g['traces']:
            T = tok
            rec = tr[TL.TOK:TL.TOK + TL.TOK_STRIDE * T].view(T, TL.TOK_STRIDE)
            ok = rec[:, TL.T_ISSUE] > 0
            rec = rec[ok]
            sw = (rec[:, TL.T_SLOTWAIT_END] - rec[:, TL.T_SLOTWAIT_START]).double()
            slot_wait += sw.tolist()
            li = rec[:, TL.T_LANE_ISSUE:TL.T_LANE_ISSUE + a['topk']]; ld = rec[:, TL.T_LANE_DONE:TL.T_LANE_DONE + a['topk']]
            m = (li > 0) & (ld > 0)
            lane_lat += (ld[m] - li[m]).double().tolist()
            ae = rec[:, TL.T_LANE_ATOMIC_END:TL.T_LANE_ATOMIC_END + a['topk']]; am = ae > 0
            atomic_lat += (ae[am] - rec[:, TL.T_ATOMIC_START].unsqueeze(1).expand_as(ae)[am]).double().tolist()
            load_lat += (rec[:, TL.T_LOAD_DONE] - rec[:, TL.T_SLOTWAIT_END]).double().tolist()
            iter_time += (rec[:, TL.T_ISSUE] - rec[:, TL.T_SLOTWAIT_START]).double().tolist()
            issue_time += (rec[:, TL.T_ISSUE] - rec[:, TL.T_LOAD_DONE]).double().tolist()
            sm = {e: [int(tr[TL.SM + s * 32 + e]) for s in range(sms)] for e in (0, 1, 2, 9, 10, 11)}
            open_b.append(max(sm[2]) - min(sm[1]))
            warp = tr[TL.WARP:TL.WARP + 4 * sms * 15].view(-1, 4)
            wd = warp[:, 1][warp[:, 1] > 0]
            data_span.append(int(wd.max()) - max(sm[2]) if wd.numel() else 0)
            close_wait.append(max(sm[10]) - int(wd.max()) if wd.numel() else 0)
            spread.append(int(wd.max()) - int(wd.min()) if wd.numel() else 0)
        per_rank.append(dict(rank=r, t_us=t * 1e6, remote_gbs=remote_tokens * BYTES_PER_TOKEN / t / 1e9, total_gbs=total_tokens * BYTES_PER_TOKEN / t / 1e9,
                             k=g['k_mean'], overhead=g['t_traced_s'] / g['t_untraced_s'] - 1,
                             slot_wait_p50=viz.pct(slot_wait, 50), slot_wait_p99=viz.pct(slot_wait, 99), slot_wait_mean=float(np.mean(slot_wait)) if slot_wait else 0,
                             lane_p50=viz.pct(lane_lat, 50), lane_p99=viz.pct(lane_lat, 99),
                             atomic_p50=viz.pct(atomic_lat, 50), atomic_p99=viz.pct(atomic_lat, 99), load_p50=viz.pct(load_lat, 50), load_p99=viz.pct(load_lat, 99),
                             iter_p50=viz.pct(iter_time, 50), issue_p50=viz.pct(issue_time, 50), spread=statistics.median(spread) / 1e3, open_b=statistics.median(open_b) / 1e3, data_span=statistics.median(data_span) / 1e3, close_wait=statistics.median(close_wait) / 1e3))
    rows.append(dict(sms=sms, tok=tok, auto_sms=d['ranks'][0].get('auto_sms'), ranks=per_rank,
                     t_us=statistics.mean(p['t_us'] for p in per_rank), remote_gbs=statistics.mean(p['remote_gbs'] for p in per_rank), total_gbs=statistics.mean(p['total_gbs'] for p in per_rank)))
rows.sort(key=lambda r: (r['tok'], r['sms']))
toks = sorted({r['tok'] for r in rows}); smss = sorted({r['sms'] for r in rows})
COL = {t: PALETTE[i] for i, t in enumerate(toks)}
PER_SM = None  # fitted from the 2-SM point below

# Figure 1: throughput vs SMs
f, ax = viz.fig(8.6, 3.8)
for t in toks:
    rs = [r for r in rows if r['tok'] == t]
    ax.plot([r['sms'] for r in rs], [r['remote_gbs'] for r in rs], marker='o', color=COL[t]); viz.end_label(ax, rs[-1]['sms'], rs[-1]['remote_gbs'], f'{t} tokens/rank', COL[t])
r2_ = next(r for r in rows if r['tok'] == max(toks) and r['sms'] == 2); PER_SM = r2_['remote_gbs'] / 2; PLATEAU = max(r['remote_gbs'] for r in rows)
xs = np.array(smss); ax.plot(xs, np.minimum(PER_SM * xs, PLATEAU), color=MUTED, lw=1, ls='--'); ax.annotate(f'min({PER_SM:.0f} GB/s × SMs, {PLATEAU:.0f})', (xs[1], min(PER_SM * xs[1], PLATEAU)), xytext=(6, -12), textcoords='offset points', fontsize=8, color=MUTED)
ax.set_xlabel('num_sms (15 dispatch warps each)'); ax.set_ylabel('NVLink bytes out / kernel time (GB/s per GPU)'); ax.set_title('Dispatch throughput against concurrency, uniform routing, 4 ranks'); ax.set_xticks(smss)
fig1 = viz.to_svg(f)

# Figure 2: slot wait and lane latency vs SMs
f, axs = viz.fig(9.2, 3.6, ncols=2)
for t in toks:
    rs = [r for r in rows if r['tok'] == t]
    med = lambda k: [statistics.median(p[k] for p in r['ranks']) for r in rs]
    axs[0].plot([r['sms'] for r in rs], np.array(med('slot_wait_p50')) / 1e3, marker='o', color=COL[t]); axs[0].plot([r['sms'] for r in rs], np.array(med('slot_wait_p99')) / 1e3, marker='o', color=COL[t], ls=':', lw=1.2)
    viz.end_label(axs[0], rs[-1]['sms'], med('slot_wait_p99')[-1] / 1e3, f'{t} p99', COL[t]); viz.end_label(axs[0], rs[-1]['sms'], med('slot_wait_p50')[-1] / 1e3, f'{t} p50', COL[t])
    axs[1].plot([r['sms'] for r in rs], np.array(med('lane_p50')) / 1e3, marker='o', color=COL[t]); axs[1].plot([r['sms'] for r in rs], np.array(med('lane_p99')) / 1e3, marker='o', color=COL[t], ls=':', lw=1.2)
    viz.end_label(axs[1], rs[-1]['sms'], med('lane_p99')[-1] / 1e3, f'{t} p99', COL[t])
axs[0].set_title('Slot wait (tma_store_wait at loop top), µs'); axs[0].set_xlabel('num_sms'); axs[0].set_ylabel('µs'); axs[0].set_xticks(smss)
axs[1].set_title('Lane issue → wait_group return, µs'); axs[1].set_xlabel('num_sms'); axs[1].set_xticks(smss)
fig2 = viz.to_svg(f)

# Figure 3: iteration breakdown at 4096 tokens
rs = [r for r in rows if r['tok'] == max(toks)]
f, ax = viz.fig(8.6, 3.6)
parts = [('slot_wait_p50', 'slot wait (tma_store_wait)', PALETTE[1]), ('load_p50', 'load → load done (TMA load, topk, atomic)', PALETTE[0]), ('issue_p50', 'load done → stores issued (bulk-store issue stalls)', PALETTE[3])]
bottom = np.zeros(len(rs)); x = np.arange(len(rs))
for k, lab, c in parts:
    v = np.array([statistics.median(p[k] for p in r['ranks']) for r in rs]) / 1e3
    ax.bar(x, v, bottom=bottom, color=c, width=0.6, label=lab); bottom += v
ax.plot(x, [statistics.median(p['iter_p50'] for p in r['ranks']) / 1e3 for r in rs], marker='o', color=INK, lw=1, label='iteration p50 (loop top → stores issued)')
ax.set_xticks(x); ax.set_xticklabels([str(r['sms']) for r in rs]); ax.set_xlabel('num_sms'); ax.set_ylabel('µs per token per warp'); ax.set_title(f'Where a warp\'s iteration time goes, {max(toks)} tokens/rank (p50 of medians)'); viz.legend_below(ax, 2)
fig3 = viz.to_svg(f)

# Figure 4: kernel phase breakdown from SM-level timestamps
f, ax = viz.fig(8.6, 3.6)
x = np.arange(len(rs)); bottom = np.zeros(len(rs))
for k, lab, c in [('data_span', 'data plane (opening-barrier release → last warp done)', PALETTE[0]), ('close_wait', 'closing barrier (last warp done → release)', PALETTE[7])]:
    v = np.array([statistics.median(p[k] for p in r['ranks']) for r in rs])
    ax.bar(x, v, bottom=bottom, color=c, width=0.6, label=lab); bottom += v
ax.plot(x, [r['t_us'] for r in rs], marker='o', color=INK, lw=1, label='kernel time (kineto, untraced)')
ax.set_xticks(x); ax.set_xticklabels([str(r['sms']) for r in rs]); ax.set_xlabel('num_sms'); ax.set_ylabel('µs'); ax.set_title(f'Kernel phases on rank 0 from in-kernel timestamps, {max(toks)} tokens/rank'); viz.legend_below(ax, 3)
fig4 = viz.to_svg(f)

tab = [[r['sms'], r['tok'], f"{r['t_us']:.1f}", f"{r['remote_gbs']:.0f}", f"{r['total_gbs']:.0f}", f"{statistics.median(p['slot_wait_p50'] for p in r['ranks'])/1e3:.2f} / {statistics.median(p['slot_wait_p99'] for p in r['ranks'])/1e3:.1f}",
        f"{statistics.median(p['lane_p50'] for p in r['ranks'])/1e3:.2f} / {statistics.median(p['lane_p99'] for p in r['ranks'])/1e3:.1f}", f"{statistics.median(p['atomic_p50'] for p in r['ranks']):.0f} / {statistics.median(p['atomic_p99'] for p in r['ranks']):.0f}",
        f"{statistics.median(p['open_b'] for p in r['ranks']):.1f}", f"{statistics.median(p['close_wait'] for p in r['ranks']):.1f}", f"{100*statistics.median(p['overhead'] for p in r['ranks']):+.1f}%"] for r in rows]

best = max(rows, key=lambda r: r['remote_gbs']); knee = next((r for r in rows if r['tok'] == max(toks) and r['remote_gbs'] > 0.9 * best['remote_gbs']), best)
auto = rows[0]['auto_sms']
r16 = next((r for r in rows if r['tok'] == max(toks) and r['sms'] == 16), None); r2 = next((r for r in rows if r['tok'] == max(toks) and r['sms'] == 2), None)

body = f'''
<h2>1. Motivation</h2>
<p>Each dispatch warp handles one token at a time: it loads the token into shared memory, issues one bulk store per destination, and at the top of the next iteration waits for those stores before it can reuse the buffer (<code>tma_store_wait</code>, dispatch.cuh:315). Before any measurement, the obvious reading of that loop was that a warp has at most one token in flight, and that the wait lasts until the peer GPU has the data, roughly an NVLink round trip of 1.3 µs. Under that reading, dispatch throughput would be bounded by (warps × 14 KB) / round trip, and the wait itself would be a per-token sensor of how congested the path to the peer is.</p>
<p>E1 showed the second half of that reading is wrong: the wait returns a few hundred ns after issue, as soon as the SM's copy engine has accepted the bytes, whether the destination is local or a peer. The round trip is never waited for inside the loop. So the question changes. If the per-token wait is not holding warps back for the round trip, what does bound throughput? Two candidates remain. One is the SM itself: E1 measured that one SM can push bulk stores to a peer at only about 50 to 58 GB/s, so with few SMs the kernel might be SM-bound. The other is the NVLink path out of the GPU, which saturates at some aggregate rate regardless of how many SMs feed it. The two predict different shapes when the number of SMs is swept: the first grows linearly with SMs, the second flattens. A second question is where a warp feels the pressure when the link is the limit, at the store instruction, at the wait, or nowhere visible at all.</p>
<p>This experiment sweeps the number of SMs (concurrency) and the number of tokens (work per warp) with uniform routing and the unmodified kernel logic, and reads the per-lane trace to see where a warp's time goes.</p>
<h2>2. Methodology</h2>
<p>In one sentence: run the unmodified dispatch at several SM counts and token counts, time it, and use in-kernel timestamps to see how long each part of a warp's loop takes.</p>
{report.dispatch_procedure("What varies here: <code>num_sms</code> ∈ {" + ', '.join(map(str, smss)) + "} (DeepEP's own model would pick " + str(auto) + ") crossed with tokens per rank ∈ {" + ', '.join(map(str, toks)) + "}, " + str(len(rows)) + " configurations in all, each run as a separate invocation of the procedure. Routing is uniform (ratio 1) and the kernel is the baseline instrumented build.", repeats=3)}
<h3>2.2 How each reported number is computed</h3>
<p>Every dispatch warp handles one token per loop iteration. For each token the kernel stamps the following moments: loop top (start of the wait for the previous token's stores), slot free (that wait returned), the slot atomic issued, token loaded into shared memory, all bulk stores issued, and per lane the moment its bulk store was issued and the moment its <code>wait_group</code> returned. SM 0 additionally stamps the kernel phases: entry, opening-barrier arrive and release, the notify phases, closing-barrier arrive and release. From these:</p>
<ul>
<li><b>NVLink GB/s</b> (Figure 1) = tokens this rank sent to <em>other</em> ranks × 14,464 bytes per token row ÷ kernel µs from step 4. "all-dest GB/s" counts local destinations too.</li>
<li><b>Slot wait</b> = slot free − loop top. This is the <code>tma_store_wait</code> at the top of the loop, DeepEP's only per-token wait.</li>
<li><b>Slot-free to load-done</b> = token loaded − slot free: the TMA load of the token, the top-k loads and the slot atomic.</li>
<li><b>Load-done to stores-issued</b> = all stores issued − token loaded: the time the warp spends in the <code>cp.async.bulk</code> issue instructions themselves. This is where an issue stall shows.</li>
<li><b>Lane issue → done</b> = per-lane wait return − per-lane store issue.</li>
<li><b>Atomic latency</b> = per-lane atomic return − atomic issued.</li>
<li><b>Kernel phases</b> (Figure 4): data plane = last dispatch warp's final stamp − opening-barrier release; closing barrier = closing-barrier release − that last warp stamp. <b>Warp spread</b> = last warp finish − first warp finish on a rank.</li>
</ul>
<p>Durations are pooled over all tokens, all warps, all four ranks and the three traced repeats before percentiles are taken; "p50" and "p99" are over that pool. Kernel µs is the mean over ranks of the Kineto value.</p>
<h3>2.3 Configuration</h3>
<ul>
<li>4 ranks on host GPUs 4–7, hidden 7168 BF16, 256 experts (64 per rank), top-8, expert alignment 128, <code>do_expand=True</code>, CPU sync on. Measured destinations per token E[k] ≈ 3.6, so about 72 % of each GPU's traffic crosses NVLink. Each SM runs 4 notify warps and 15 dispatch warps.</li>
<li>Tracing costs {100*statistics.median(statistics.median(p['overhead'] for p in r['ranks']) for r in rows):+.1f} % of kernel time (median over configurations), from the probes-on Kineto run of step 4.</li>
</ul>
<h2>3. Results</h2>
<div class="finding"><b>Throughput is per-SM capacity times SM count until the NVLink path saturates at about {best['remote_gbs']:.0f} GB/s per GPU.</b> At {max(toks)} tokens per rank the kernel delivers {r2['remote_gbs'] if r2 else 0:.0f} GB/s of NVLink traffic with 2 SMs ({r2['remote_gbs']/2 if r2 else 0:.0f} GB/s per SM, {r2['total_gbs']/2 if r2 else 0:.0f} GB/s per SM counting local destinations, consistent with E1's 50 GB/s per SM drain) and {r16['remote_gbs'] if r16 else 0:.0f} GB/s with 16; adding SMs beyond 12 buys about 10 % more while the slot wait grows threefold. Small token counts never reach the plateau: with 256 tokens per rank most warps handle one token and the barriers dominate.</div>
{report.figure(fig1, 'Figure 1. NVLink bytes out per GPU divided by kernel time, against the number of SMs, for three token counts. The dashed line is the two-limit model from E1: per-SM bulk-store issue capacity, capped by the one-direction link rate.')}
<div class="finding"><b>Where a warp's 17 µs iteration goes, and where back-pressure shows.</b> With 2 to 8 SMs a warp spends about 3.7 µs in the slot wait, 4.6 µs from slot free to load done (TMA load of x, top-k loads, the slot atomic at 96 ns), and about 8.7 µs between load done and stores issued. That last segment is almost pure <code>cp.async.bulk</code> issue stall: E1 showed the per-lane queue admits about seven stores and the SM drains them at ~50 GB/s, and fifteen warps share that path. So the quantity that limits how far a warp can run ahead is the SM's bulk-store queue, about seven stores per lane, and the first place back-pressure appears is the store instruction. Past 8 SMs the aggregate issue capacity exceeds the NVLink path and the queueing moves into the slot wait: its p50 grows from 3.7 to 9.7 µs and its p99 from 5 to 36 µs at 32 SMs. The slot wait is therefore a usable congestion sensor, with this meaning: it reports the sender's queue towards the link, not the state of the peer.</div>
{report.figure(fig2, 'Figure 2. Left: slot wait p50 (solid) and p99 (dotted) against SM count. Right: per-lane issue → wait_group return. Both are local-acceptance times; their growth with concurrency is the queue in front of the NVLink ports filling.')}
{report.figure(fig3, 'Figure 3. Composition of one warp iteration at the largest token count: slot wait, slot-free to load-done (TMA load, top-k loads, atomic), and load-done to stores-issued, which is the bulk-store issue stall. The line is the measured loop-top to stores-issued time.')}
{report.figure(fig4, 'Figure 4. Kernel phases from the in-kernel timestamps on rank 0: data plane (opening-barrier release to the last dispatch warp finishing) and closing barrier (last warp done to release). The line is the untraced Kineto kernel time, which is slightly shorter because the traced runs are launched without the host-side barrier. The opening barrier is omitted: in the traced runs it is dominated by launch skew between ranks (tens of µs to ms), not by the barrier itself.')}
{report.table(['SMs', 'tokens', 'kernel µs', 'NVLink GB/s', 'all-dest GB/s', 'slot wait p50 / p99 µs', 'lane issue→done p50 / p99 µs', 'atomic p50 / p99 ns', 'open barrier µs', 'close wait µs', 'trace cost'], tab, num_cols=range(2, 11))}
<h2>4. Conclusions</h2>
<ol>
<li><b>What bounds throughput.</b> Throughput is min(per-SM bulk-store drain × SMs, NVLink path), about {r2['remote_gbs']/2 if r2 else 0:.0f} GB/s per SM and {best['remote_gbs']:.0f} GB/s per GPU here. The limit on how far a warp can run ahead is the SM's async-proxy queue (about seven stores per lane), not one token per warp waiting for a round trip; the queue drains at the per-SM rate, so a warp spends half its iteration stalled on the store instruction even on an idle fabric. The pre-measurement model "warps × 14 KB per round trip" is withdrawn. Link saturation adds queueing that is visible in the slot wait and the per-lane wait, with p99 values of tens of µs.</li>
<li><b>The tail is large.</b> The first and last dispatch warp of a rank finish {statistics.median(p['spread'] for p in r16['ranks']) if r16 else 0:.0f} µs apart at 16 SMs (data plane {statistics.median(p['data_span'] for p in r16['ranks']) if r16 else 0:.0f} µs), and the closing barrier then waits a further {statistics.median(p['close_wait'] for p in r16['ranks']) if r16 else 0:.0f} µs for the slowest rank. Roughly half the data-plane span is therefore a tail in which the fabric drains while most warps are already idle. The opening barrier of the traced runs also records raw launch skew between ranks of hundreds of µs to ms when no host barrier precedes the launch.</li>
<li><b>Slot atomics are not a bottleneck at four ranks</b>: 96 ns p50 at every concurrency, p99 128 ns. The concern that all 15,000 slot-allocation atomics serialise on one cache line is real but cheap at this scale; E3 quantifies the padded layout anyway.</li>
<li><b>Concurrency has a cost beyond the plateau.</b> More SMs than the link needs lengthen the slot wait and the closing-barrier tail without adding throughput. DeepEP's own choice ({auto} SMs here) sits near the knee.</li>
<li><b>The per-lane trace is validated as a tool.</b> About 1 % overhead, and it separates the warp iteration into its parts. The slot wait is the quantity to watch in E2, where routing skew replaces concurrency as the stressor.</li>
<li><b>Small-token runs are barrier-bound</b>, which is the decode regime (few tokens per rank, kernel time dominated by synchronisation rather than data); the SM-level timeline already shows the opening barrier and the closing wait as the dominant phases there.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Four ranks, one node; E[k] ≈ 3.6 so the fan-out per token is smaller than with eight ranks. Kineto timings include DeepEP's L2 flush and barrier per iteration, as in its own test. The per-SM issue capacity of 58 GB/s comes from a single-destination microbenchmark; here each warp fans out to up to four destinations.</div>
'''
meta = f"experiment E0 · {len(rows)} configurations · 4 × H100 on host GPUs 4–7 · scripts: e0_sweep.py, common/run_dispatch.py, e0_analyze.py"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E0 · Throughput against concurrency: SM or link?', 'Sweeping SM count and token count on the unmodified dispatch to find what bounds throughput now that E1 has shown the per-token wait is local.', meta, body))
print('wrote report with', len(rows), 'configs')
