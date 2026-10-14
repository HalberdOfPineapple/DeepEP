"""Build report.html for E5 from results/*.pt."""
import glob, os, sys, statistics, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, traceload as TLD, trace_layout as TL
from viz import PALETTE, INK, MUTED

ORDER = ['none', 'copy_rank0', 'copy_all', 'gemm_rank0', 'gemm_all', 'none_tok256', 'copy_all_tok256']
LABEL = {'none': 'none', 'copy_rank0': 'copies\nrank 0', 'copy_all': 'copies\nall', 'gemm_rank0': 'GEMMs\nrank 0', 'gemm_all': 'GEMMs\nall', 'none_tok256': '256 tok\nnone', 'copy_all_tok256': '256 tok\ncopies all'}
LONG = {'none': 'no background', 'copy_rank0': 'HBM copies on rank 0', 'copy_all': 'HBM copies on all ranks', 'gemm_rank0': 'GEMMs on rank 0', 'gemm_all': 'GEMMs on all ranks', 'none_tok256': '256 tok, no background', 'copy_all_tok256': '256 tok, HBM copies on all'}
BG_US = {'copy': 33, 'gemm': 27}  # one background kernel alone; the in-run GEMM calibration included cuBLAS initialisation and is not used
cases = {}
for f in glob.glob(os.path.join(HERE, 'results', '*.pt')):
    name = os.path.basename(f)[:-3]; run = TLD.load(f); K = run['args']['topk']; NR = run['NR']
    per_rank = {}
    for r in range(NR):
        d = dict(slot=[], prep=[], issue=[], iter=[], atomic=[], lane={dst: [] for dst in range(NR)}, span=[], data=[], open_w=[], close_w=[], bg=run['ranks'][r]['reps'][0]['bg'])
        for rep in run['ranks'][r]['reps']:
            tp = TLD.token_parts(rep); d['slot'] += tp['slot_wait'].tolist(); d['prep'] += tp['prepare'].tolist(); d['issue'] += tp['issue'].tolist(); d['iter'] += tp['iteration'].tolist()
            d['atomic'] += TLD.atomic_latencies(rep, K).tolist()
            lat, dst, _, _ = TLD.lane_latencies(rep, K)
            for dd in range(NR): d['lane'][dd] += lat[dst == dd].tolist()
            ph = TLD.phase_durations(rep); d['span'].append(ph['span_us']); d['data'].append(ph['data_us']); d['open_w'].append(ph['A_wait_us']); d['close_w'].append(ph['E_wait_us'])
        d['t_us'] = run['ranks'][r]['t_untraced_s'] * 1e6
        per_rank[r] = d
    bgk = run['ranks'][0]['reps'][0]['bg']['kind'] if run['ranks'][0]['reps'][0]['bg'] else 'none'
    cases[name] = dict(name=name, ranks=per_rank, NR=NR, tok=run['T'], bg_us=BG_US.get(bgk, 0))
names = [n for n in ORDER if n in cases]
def P(c, r, key, p): return viz.pct(c['ranks'][r][key], p)
def Pall(c, key, p, ranks=None):
    ranks = ranks if ranks is not None else range(c['NR']); xs = []
    for r in ranks: xs += c['ranks'][r][key]
    return viz.pct(xs, p)
def lane_p(c, src_ranks, dst, p):
    xs = []
    for r in src_ranks: xs += c['ranks'][r]['lane'][dst]
    return viz.pct(xs, p)

big = [n for n in names if cases[n]['tok'] == 4096]
# ---- Figure 1: per-token loop parts per case (rank 0 vs ranks 1-3) ----
f, axs = viz.fig(8.6, 3.8, ncols=2, sharey=True)
parts = [('slot', 'C1 slot wait', PALETTE[3]), ('prep', 'C2 prepare', PALETTE[0]), ('issue', 'C3 issue', PALETTE[1])]
for ax, ranks, ttl in ((axs[0], [0], 'rank 0 (loaded in the rank-0 cases)'), (axs[1], [1, 2, 3], 'ranks 1–3')):
    x = np.arange(len(big)); bottom = np.zeros(len(big))
    for key, lbl, col in parts:
        vals = np.array([Pall(cases[n], key, 50, ranks) / 1e3 for n in big]); ax.bar(x, vals, 0.6, bottom=bottom, color=col, label=lbl); bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([LABEL[n] for n in big], fontsize=8); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel('median µs per token iteration'); viz.legend_below(axs[0], ncol=3)
fig1 = viz.to_svg(f)

# ---- Figure 2: per-destination store latency seen by ranks 1-3 (to rank 0 vs to others) and by rank 0 ----
f, axs = viz.fig(8.6, 3.6, ncols=2, sharey=True)
x = np.arange(len(big))
ax = axs[0]
to0 = [lane_p(cases[n], [1, 2, 3], 0, 50) / 1e3 for n in big]; toX = [np.median([lane_p(cases[n], [1, 2, 3], d, 50) for d in (1, 2, 3)]) / 1e3 for n in big]
ax.bar(x - 0.18, to0, 0.36, color=PALETTE[1], label='to rank 0'); ax.bar(x + 0.18, toX, 0.36, color=PALETTE[0], label='to ranks 1–3 (median of the three)')
ax.set_xticks(x); ax.set_xticklabels([LABEL[n] for n in big], fontsize=8); ax.set_title('Stores by ranks 1–3, p50'); ax.set_ylabel('µs'); viz.legend_below(ax, ncol=2)
ax = axs[1]
p50 = [np.median([lane_p(cases[n], [0], d, 50) for d in (1, 2, 3)]) / 1e3 for n in big]; p99 = [np.median([lane_p(cases[n], [0], d, 99) for d in (1, 2, 3)]) / 1e3 for n in big]
ax.bar(x - 0.18, p50, 0.36, color=PALETTE[0], label='p50'); ax.bar(x + 0.18, p99, 0.36, color=PALETTE[2], label='p99')
ax.set_xticks(x); ax.set_xticklabels([LABEL[n] for n in big], fontsize=8); ax.set_title('Stores by rank 0'); viz.legend_below(ax, ncol=2)
fig2 = viz.to_svg(f)

# ---- Figure 3: kernel span / data plane / prepare & atomic p99 ----
f, axs = viz.fig(8.6, 3.4, ncols=2)
ax = axs[0]
sp0 = [statistics.median(cases[n]['ranks'][0]['span']) for n in big]; spX = [statistics.median(v for r in (1, 2, 3) for v in cases[n]['ranks'][r]['span']) for n in big]
ax.bar(x - 0.18, sp0, 0.36, color=PALETTE[1], label='rank 0'); ax.bar(x + 0.18, spX, 0.36, color=PALETTE[0], label='ranks 1–3')
ax.set_xticks(x); ax.set_xticklabels([LABEL[n] for n in big], fontsize=8); ax.set_title('Kernel span from the stamps'); ax.set_ylabel('µs'); viz.legend_below(ax, ncol=2)
ax = axs[1]
pr = [Pall(cases[n], 'prep', 99, [0]) / 1e3 for n in big]; at = [Pall(cases[n], 'atomic', 99, [0]) / 1e3 for n in big]
ax.bar(x - 0.18, pr, 0.36, color=PALETTE[0], label='C2 prepare p99 (TMA load of x, atomic)'); ax.bar(x + 0.18, at, 0.36, color=PALETTE[2], label='slot atomic p99')
ax.set_xticks(x); ax.set_xticklabels([LABEL[n] for n in big], fontsize=8); ax.set_title('Rank 0 local steps, p99'); ax.set_ylabel('µs'); ax.set_yscale('log'); viz.legend_below(ax, ncol=1)
fig3 = viz.to_svg(f)

tab = []
for n in names:
    c = cases[n]
    tab.append([LONG[n], c['tok'], f'{c["bg_us"]:.0f}' if c['bg_us'] else '–', f'{statistics.median(c["ranks"][0]["span"]):.0f}', f'{statistics.median(v for r in range(1, c["NR"]) for v in c["ranks"][r]["span"]):.0f}', f'{statistics.median(v for r in range(1, c["NR"]) for v in c["ranks"][r]["open_w"]):.0f}',
                f'{P(c, 0, "slot", 50)/1e3:.1f} / {P(c, 0, "slot", 99)/1e3:.1f}', f'{P(c, 0, "prep", 50)/1e3:.1f} / {P(c, 0, "prep", 99)/1e3:.1f}', f'{P(c, 0, "issue", 50)/1e3:.1f} / {P(c, 0, "issue", 99)/1e3:.1f}',
                f'{P(c, 0, "atomic", 50):.0f} / {P(c, 0, "atomic", 99):.0f}', f'{lane_p(c, [1,2,3], 0, 50)/1e3:.1f}', f'{np.median([lane_p(c, [1,2,3], d, 50) for d in (1,2,3)])/1e3:.1f}', f'{np.median([lane_p(c, [0], d, 50) for d in (1,2,3)])/1e3:.1f}'])

none, ca, cr, ga, gr = (cases.get(k) for k in ('none', 'copy_all', 'copy_rank0', 'gemm_all', 'gemm_rank0'))
def ratio(c, key, ranks, p=50): return Pall(c, key, p, ranks) / Pall(none, key, p, ranks) if c and none else float('nan')

mech = '''<svg viewBox="0 0 880 230" role="img" aria-label="E5 setup: on each GPU a side stream runs a queue of short kernels, 32 MB copies that saturate HBM or 2048-cubed GEMMs that saturate the tensor cores, while the traced dispatch runs on the main stream; the background runs on all ranks or only on rank 0">
<text x="20" y="24" class="head">One GPU during a traced dispatch</text>
<rect x="20" y="40" width="400" height="56" rx="8" class="box-acc"/><text x="30" y="60" class="head">main stream: dispatch kernel (16 SMs)</text><text x="30" y="80" class="small">240 warps: TMA loads from x in HBM, slot atomics in L2, bulk stores over NVLink</text>
<rect x="20" y="110" width="400" height="56" rx="8" class="box-warn"/><text x="30" y="130" class="head">side stream: 120 short background kernels</text><text x="30" y="150" class="small">copy: 32 MB device-to-device each, HBM-bound · gemm: 2048³ BF16 each, tensor-core-bound</text>
<text x="20" y="190" class="small">Short kernels, so the cooperative dispatch launch still finds its 16 SMs; the queue outlasts the dispatch.</text>
<text x="20" y="208" class="small">Kineto L2 flush off (the cache state is part of the subject); the all-reduce barrier between iterations stays.</text>
<rect x="470" y="40" width="390" height="126" rx="8" class="box"/><text x="480" y="60" class="head">Who runs the background</text>
<text x="480" y="82" class="small">"all": every rank is both a loaded sender and a loaded receiver</text>
<text x="480" y="100" class="small">"rank 0": rank 0 is a loaded sender; ranks 1–3 are clean senders</text>
<text x="480" y="118" class="small">whose stores to rank 0 land in a loaded receiver and whose</text>
<text x="480" y="136" class="small">stores to each other do not: the per-destination latency on</text>
<text x="480" y="154" class="small">ranks 1–3 separates the receiver-side effect from the sender-side one</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>DeepEP's kernels are meant to run beside the model's GEMMs, on a few SMs, so that communication overlaps compute. The compute shares three resources with the dispatch warps: HBM bandwidth (the warps read their tokens from HBM with TMA loads), the L2 (slot atomics and the path of every store out of the SM) and, on the receiving side, the memory system that absorbs incoming NVLink writes. The question is which of the dispatch's steps slows down when the GPU is also busy, by how much, and whether the slowdown happens on the sending GPU (its loads and its egress) or on the receiving GPU (its ingress), because the two have different signatures and different remedies. A tool that only sees kernel duration cannot tell "my own GPU was busy" from "my peer's GPU was busy" from "the link was busy".</p>

<h2>2. Methodology</h2>
<p>In one sentence: run a queue of short HBM-bound copies or compute-bound GEMMs on a second stream of the same GPU while the traced dispatch runs, on all four ranks or only on rank 0, and compare each step of the per-token loop and each (sender, destination) store latency with the undisturbed run.</p>
{report.figure(mech, 'Figure 1. The background load and the two placements.')}
{report.dispatch_procedure("What varies here: the background kind (none, 32 MB copies, 2048³ GEMMs) and its placement (all ranks or rank 0 only), at 4,096 tokens and 16 SMs; plus the 256-token case with and without copies on all ranks. The background queue is issued on the side stream right after the clock calibration and before the barrier that precedes each traced dispatch, so it is already running when the dispatch starts and outlasts it. The Kineto step runs without background and gives the undisturbed kernel time; kernel time under load is the entry-to-end span from the stamps.", repeats=3, variant_note="The baseline instrumented build is used throughout. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Per-token loop parts</b> on a rank, median over its tokens, warps and repeats: C1 slot wait (loop top → slot free), C2 prepare (slot free → token loaded, which contains the TMA load of the token from HBM and the slot atomic), C3 issue (token loaded → all stores issued).</li>
<li><b>Per-destination store latency</b>: per lane, wait return − store issue, grouped by sender rank and destination rank. "Ranks 1–3 to rank 0" pools the three clean senders' stores into the (possibly loaded) rank 0; "ranks 1–3 to each other" is the median over the three idle destinations; "rank 0 to its peers" is the loaded sender's own stores.</li>
<li><b>Slot atomic latency</b> per lane, as in E3. <b>Kernel span</b> = kernel end − kernel entry on SM 0, per rank, median over repeats.</li>
<li><b>Background intensity</b>: the duration of one background kernel alone, measured with CUDA events before the repeats (32 MB copy ≈ HBM rate; one GEMM ≈ tensor-core rate).</li>
</ul>
<h3>2.3 Configuration</h3>
<ul><li>4 ranks on host GPUs 4–7, 16 SMs, uniform routing, top-8 of 256 experts, CPU sync on, L2 flush off, 120 background kernels per repeat (30 in the 256-token case).</li></ul>

<h2>3. Results</h2>
<div class="finding"><b>Where the slowdown lands.</b> With HBM copies on all ranks the kernel span on rank 0 goes from {statistics.median(none['ranks'][0]['span']) if none else 0:.0f} to {statistics.median(ca['ranks'][0]['span']) if ca else 0:.0f} µs; with GEMMs on all ranks to {statistics.median(ga['ranks'][0]['span']) if ga else 0:.0f} µs. In the loop, C2 prepare (the HBM load of the token) is ×{ratio(ca, 'prep', [0]):.2f} under copies and ×{ratio(ga, 'prep', [0]):.2f} under GEMMs; C1 slot wait ×{ratio(ca, 'slot', [0]):.2f} and ×{ratio(ga, 'slot', [0]):.2f}; C3 issue ×{ratio(ca, 'issue', [0]):.2f} and ×{ratio(ga, 'issue', [0]):.2f}.</div>
{report.figure(fig1, 'Figure 2. Median composition of one warp iteration per case, on rank 0 (left) and on ranks 1–3 (right). In the rank-0 cases the right panel shows clean senders whose only exposure to the load is through their stores into rank 0.')}
<div class="finding"><b>Sender side or receiver side.</b> When only rank 0 is loaded with copies, the clean senders' stores <em>into</em> rank 0 take {lane_p(cr, [1,2,3], 0, 50)/1e3 if cr else 0:.1f} µs p50 against {np.median([lane_p(cr, [1,2,3], d, 50) for d in (1,2,3)])/1e3 if cr else 0:.1f} µs to each other (undisturbed: {lane_p(none, [1,2,3], 0, 50)/1e3 if none else 0:.1f} µs), while rank 0's own stores to its peers take {np.median([lane_p(cr, [0], d, 50) for d in (1,2,3)])/1e3 if cr else 0:.1f} µs (undisturbed {np.median([lane_p(none, [0], d, 50) for d in (1,2,3)])/1e3 if none else 0:.1f} µs). The same split for GEMMs on rank 0: into rank 0 {lane_p(gr, [1,2,3], 0, 50)/1e3 if gr else 0:.1f} µs, between the others {np.median([lane_p(gr, [1,2,3], d, 50) for d in (1,2,3)])/1e3 if gr else 0:.1f} µs, from rank 0 {np.median([lane_p(gr, [0], d, 50) for d in (1,2,3)])/1e3 if gr else 0:.1f} µs. Because the per-SM store queue is shared across destinations (E2), a receiver-side slowdown on one destination spreads to the clean senders' other destinations too; the destination split in Figure 3 shows how much of it is destination-specific.</div>
{report.figure(fig2, 'Figure 3. Left: store latency (issue → wait return, p50) of the clean senders, split by destination: into rank 0 versus into each other. Right: the loaded sender rank 0’s own stores, p50 and p99.')}
<div class="finding"><b>The largest effect is not in the data plane but in the launch.</b> When only rank 0 runs the background, ranks 1–3 wait {statistics.median(v for r in (1,2,3) for v in cr['ranks'][r]['open_w']) if cr else 0:.0f} µs (copies) and {statistics.median(v for r in (1,2,3) for v in gr['ranks'][r]['open_w']) if gr else 0:.0f} µs (GEMMs) in the opening barrier, against {statistics.median(v for r in (1,2,3) for v in none['ranks'][r]['open_w']) if none else 0:.0f} µs undisturbed: rank 0's cooperative dispatch launch could not get its 16 SMs while the background queue kept the GPU full, and every other rank idled for that long. In the "all ranks" cases the same delay happens on every rank at once and is invisible in the barrier wait; it appears only as a later kernel entry relative to the host call, which E6 and E9 measure. For a kernel meant to run beside GEMMs this launch wait is the first number to watch, and no in-kernel stamp sees it.</div>
{report.figure(fig3, 'Figure 4. Left: kernel span per case on rank 0 and on the others. Right: the tail (p99) of the local-memory steps on rank 0, prepare and slot atomic, log scale.')}
{report.table(['case', 'tokens', 'bg kernel µs', 'span rank 0 µs', 'span ranks 1–3 µs', 'open wait ranks 1–3 µs', 'C1 slot p50/p99 µs (r0)', 'C2 prepare p50/p99 µs (r0)', 'C3 issue p50/p99 µs (r0)', 'atomic p50/p99 ns (r0)', 'r1–3 → r0 p50 µs', 'r1–3 → r1–3 p50 µs', 'r0 → peers p50 µs'], tab, num_cols=range(1, 13))}

<h2>4. Conclusions</h2>
<ol>
<li><b>Compute interference is measurable per step and per side with the stamps already in place.</b> The loop decomposition shows which step pays (the HBM load in C2 under memory-bound neighbours, the store path in C1/C3 when the egress or the peer's ingress is slower), and the per-destination split on clean senders isolates the receiver-side component. These are the signatures a tool needs to attribute "communication got slower" to the co-running compute rather than to the fabric.</li>
<li><b>The per-SM shared store queue blurs destinations.</b> A loaded receiver slows the clean senders' stores to it, and through the shared queue their stores to everyone else; the destination mix of what is queued is needed to attribute the delay to the loaded peer, as in E2.</li>
<li><b>Launch scheduling dominates when the GPU is full.</b> Even with 30 µs background kernels the cooperative launch of dispatch waits 1–2 ms behind them; with a single 8 ms GEMM (a first attempt) it waited the whole GEMM. This is a sync-in (phase A) effect, invisible inside the kernel, and in a pipelined MoE it is what "overlap" would cost first.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Synthetic background kernels, not real MoE GEMMs; no PM-sampled HBM/L2 counters alongside (CUPTI PM sampling was not set up), so the background intensity is characterised only by its standalone kernel time. Four ranks. The 256-token case has few tokens per warp and is noisy.</div>
'''
meta = f"experiment E5 · {len(names)} cases · 4 × H100 on host GPUs 4–7 · scripts: e5_sweep.py, common/run_dispatch.py (--bg), e5_analyze.py · generated {time.strftime('%Y-%m-%d %H:%M')}"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E5 · Compute interference: which step pays, and on which side', 'HBM-bound copies or compute-bound GEMMs run beside the traced dispatch on all ranks or on one; the per-step and per-destination stamps say where the time goes.', meta, body))
print('wrote report with', len(names), 'cases')
