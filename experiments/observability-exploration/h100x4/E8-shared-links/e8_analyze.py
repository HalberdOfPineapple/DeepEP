"""Build report.html for E8 from results/*.pt."""
import glob, os, sys, statistics, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, traceload as TLD, trace_layout as TL
from viz import PALETTE, INK, MUTED

ORDER = ['alone_nonvml', 'combine_nonvml', 'nccl6_nonvml', 'nccl16_nonvml', 'alone', 'combine', 'nccl6', 'nccl16']
LABEL = {'alone_nonvml': 'dispatch alone', 'combine_nonvml': 'dispatch + combine\n(second buffer)', 'nccl6_nonvml': 'dispatch + NCCL\nall-to-all ×6', 'nccl16_nonvml': 'dispatch + NCCL\nall-to-all ×16',
         'alone': 'alone\n+ NVML read', 'combine': 'combine\n+ NVML read', 'nccl6': 'NCCL ×6\n+ NVML read', 'nccl16': 'NCCL ×16\n+ NVML read'}
TIMING = ['alone_nonvml', 'combine_nonvml', 'nccl6_nonvml', 'nccl16_nonvml']   # timing comparisons: runs without the NVML read
COUNTERS = ['alone', 'combine', 'nccl6', 'nccl16']                             # per-link bytes: runs with the NVML read
cases = {}
for f in glob.glob(os.path.join(HERE, 'results', '*.pt')):
    name = os.path.basename(f)[:-3]; run = TLD.load(f); K = run['args']['topk']; NR = run['NR']
    d = dict(slot=[], issue=[], lane=[], span=[], data=[], tail=[], open_w=[], close_w=[], ov_alone=run['ranks'][0]['reps'][0]['overlap']['alone_ms'], ov_over=[], links=[])
    for r in range(NR):
        for rep in run['ranks'][r]['reps']:
            tp = TLD.token_parts(rep); d['slot'] += tp['slot_wait'].tolist(); d['issue'] += tp['issue'].tolist()
            lat, dst, _, _ = TLD.lane_latencies(rep, K); d['lane'] += lat.tolist()
            ph = TLD.phase_durations(rep); d['span'].append(ph['span_us']); d['data'].append(ph['data_us']); d['tail'].append(ph['tail_us']); d['open_w'].append(ph['A_wait_us']); d['close_w'].append(ph['E_wait_us'])
    for g in run['raw']['ranks']:
        d['ov_over'] = g['overlap']['overlapped_ms'] if g['rank'] == 0 else d['ov_over']
    # per-link bytes from the NVML readings bracketing each repeat (rank 0 holds them for all visible GPUs)
    for rep in run['ranks'][0]['reps']:
        s = rep['nvml']
        if not s: continue
        a, b = s[0][1], s[-1][1]
        per_gpu = {}
        for gpu in a:
            tx = np.array(b[gpu][0]) - np.array(a[gpu][0]); rx = np.array(b[gpu][1]) - np.array(a[gpu][1]); per_gpu[gpu] = (tx * 1024 / 1e6, rx * 1024 / 1e6)  # MB
        d['links'].append(per_gpu)
    cases[name] = d
names = [n for n in TIMING if n in cases]; cnames = [n for n in COUNTERS if n in cases]; allnames = [n for n in ORDER if n in cases]
def P(n, key, p): return viz.pct(cases[n][key], p)

# ---- Figure 1: dispatch metrics alone vs overlapped ----
f, axs = viz.fig(8.6, 3.6, ncols=3)
x = np.arange(len(names))
SHORT = {'alone_nonvml': 'alone', 'combine_nonvml': '+ combine', 'nccl6_nonvml': '+ NCCL ×6', 'nccl16_nonvml': '+ NCCL ×16'}
ax = axs[0]; ax.bar(x, [statistics.median(cases[n]['data']) for n in names], 0.6, color=PALETTE[0]); ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8, rotation=30); ax.set_title('Dispatch data plane'); ax.set_ylabel('µs')
ax = axs[1]; ax.bar(x - 0.18, [P(n, 'slot', 50) / 1e3 for n in names], 0.36, color=PALETTE[3], label='p50'); ax.bar(x + 0.18, [P(n, 'slot', 99) / 1e3 for n in names], 0.36, color=PALETTE[1], label='p99'); ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8, rotation=30); ax.set_title('C1 slot wait'); ax.set_ylabel('µs'); viz.legend_below(ax, ncol=2)
ax = axs[2]; ax.bar(x - 0.18, [P(n, 'lane', 50) / 1e3 for n in names], 0.36, color=PALETTE[3], label='p50'); ax.bar(x + 0.18, [P(n, 'lane', 99) / 1e3 for n in names], 0.36, color=PALETTE[1], label='p99'); ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8, rotation=30); ax.set_title('Per-lane store wait'); viz.legend_below(ax, ncol=2)
fig1 = viz.to_svg(f)

# ---- Figure 2: the other kernel's time, alone vs overlapped ----
f, ax = viz.fig(8.6, 3.0)
ov_names = [n for n in names if n != 'alone']
ax.bar(np.arange(len(ov_names)) - 0.18, [cases[n]['ov_alone'] * 1e3 for n in ov_names], 0.36, color=PALETTE[0], label='alone')
ax.bar(np.arange(len(ov_names)) + 0.18, [statistics.median(cases[n]['ov_over']) * 1e3 if cases[n]['ov_over'] else 0 for n in ov_names], 0.36, color=PALETTE[1], label='while dispatch runs')
ax.set_xticks(np.arange(len(ov_names))); ax.set_xticklabels([LABEL[n].replace('dispatch + ', '') for n in ov_names], fontsize=8); ax.set_ylabel('µs'); ax.set_title('The other kernel: combine (one call) or the NCCL all-to-all queue (whole queue)'); viz.legend_below(ax, ncol=2)
fig2 = viz.to_svg(f)

# ---- Figure 3: per-link TX bytes on GPU 4 (rank 0) per case ----
f, axs = viz.fig(8.6, 3.4, ncols=len(cnames), sharey=True)
axs = np.atleast_1d(axs)
for ax, n in zip(axs, cnames):
    L = cases[n]['links']
    if not L: ax.set_title(LABEL[n]); continue
    gpu0 = sorted(L[0])[0]
    tx = np.median([pg[gpu0][0] for pg in L], axis=0); rx = np.median([pg[gpu0][1] for pg in L], axis=0)
    ax.bar(np.arange(18) - 0.2, tx, 0.4, color=PALETTE[0], label='TX'); ax.bar(np.arange(18) + 0.2, rx, 0.4, color=PALETTE[2], label='RX')
    ax.set_title(LABEL[n].replace('\n+ NVML read', ''), fontsize=9); ax.set_xlabel('link'); ax.set_xticks([0, 5, 10, 15])
axs[0].set_ylabel('MB per traced repeat, GPU 4'); viz.legend_below(axs[0], ncol=2)
fig3 = viz.to_svg(f)

tab = []
for n in allnames:
    c = cases[n]; L = c['links']
    tot_tx = statistics.median(sum(sum(pg[g][0]) for g in pg) for pg in L) if L else float('nan')
    per_link_cv = statistics.median(float(np.std(pg[sorted(pg)[0]][0]) / max(1e-9, np.mean(pg[sorted(pg)[0]][0]))) for pg in L) if L else float('nan')
    tab.append([LABEL[n].replace('\n', ' '), f'{statistics.median(c["span"]):.0f}', f'{statistics.median(c["data"]):.0f}', f'{statistics.median(c["tail"]):.0f}', f'{P(n, "slot", 50)/1e3:.1f} / {P(n, "slot", 99)/1e3:.1f}', f'{P(n, "issue", 50)/1e3:.1f}', f'{P(n, "lane", 50)/1e3:.1f} / {P(n, "lane", 99)/1e3:.1f}',
                f'{c["ov_alone"]*1e3:.0f}' if c['ov_alone'] else '–', f'{statistics.median(c["ov_over"])*1e3:.0f}' if c['ov_over'] else '–', f'{tot_tx:.0f}', f'{per_link_cv*100:.0f}'])

al, cb, n6, n16 = (cases.get(k) for k in TIMING); al_nv = cases.get('alone')
def md(c, k): return statistics.median(c[k]) if c else float('nan')

mech = '''<svg viewBox="0 0 880 240" role="img" aria-label="E8 setup: the traced dispatch on buffer A runs on the main stream while, on a side stream, either DeepEP combine on a second buffer B or a queue of NCCL all-to-all collectives runs; both use the same NVLink ports; NVML per-link counters are read before and after each repeat">
<text x="20" y="24" class="head">One GPU, one traced repeat</text>
<rect x="20" y="40" width="380" height="50" rx="8" class="box-acc"/><text x="30" y="60" class="head">main stream: dispatch (buffer A, 16 SMs, traced)</text><text x="30" y="78" text-anchor="start" class="small">161 MB out per GPU over NVLink, 4,096 tokens</text>
<rect x="20" y="104" width="380" height="50" rx="8" class="box-warn"/><text x="30" y="124" class="head">side stream, one of:</text><text x="30" y="142" class="small">combine on buffer B (16 SMs, ~160 MB) · 6 or 16 NCCL all-to-all of 64 MB (48 MB remote each)</text>
<rect x="440" y="40" width="420" height="114" rx="8" class="box"/><text x="450" y="60" class="head">shared</text>
<text x="450" y="80" class="small">the 18 NVLink ports of the GPU and the four NVSwitch chips: both kernels' bytes cross them</text>
<text x="450" y="98" class="small">the L2 and the async-proxy queues of the SMs each kernel runs on (disjoint SM sets)</text>
<text x="450" y="124" class="small">NVML per-link TX/RX byte counters read before and after each repeat (rank 0, all four GPUs)</text>
<text x="450" y="142" class="small">CUDA events on the side stream time the other kernel alone and overlapped</text>
<text x="20" y="185" class="small">Combine on a second EPBuffer: a separate symmetric window and its own barriers, so the two DeepEP kernels do not share state; they share links, SMs' L2 slices and the switch.</text>
<text x="20" y="203" class="small">The dispatch's handle for buffer B comes from one untraced dispatch before the repeats; the same combine is replayed every repeat.</text>
<text x="20" y="225" class="small">NCCL all-to-all through a second process group, async, 6 or 16 back to back so that the queue outlasts the dispatch.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>In a pipelined MoE layer the dispatch of one micro-batch runs while the combine of another is still in flight, and both share the GPU's NVLink ports with whatever else the process, or another process on the same GPU, is sending. When two kernels share links, each one's duration grows, but neither has any way to see the other: the dispatch's own stamps show stores taking longer, not why. The question is what the sharing costs each kernel, whether the per-lane stamps of the traced kernel change in a way that distinguishes "another kernel is on my links" from "my own fan-out saturates them" (E0) or "one destination is hot" (E2), and whether link-level counters outside the kernel can supply the missing attribution.</p>

<h2>2. Methodology</h2>
<p>In one sentence: run the traced dispatch alone, then with a DeepEP combine on a second buffer running at the same time, then with a queue of NCCL all-to-all collectives running at the same time, and compare the dispatch's stamps, the other kernel's duration, and the per-link byte counters.</p>
{report.figure(mech, 'Figure 1. The two kernels and what they share.')}
{report.dispatch_procedure("What varies here: what runs on the side stream during each traced dispatch: nothing; a DeepEP combine on a second buffer; six NCCL all-to-all calls of 64 MB; sixteen of them. The side-stream work is enqueued right before the barrier that precedes the dispatch, so it is already running when the dispatch starts. NVML link counters are read by rank 0 on all four GPUs before and after each repeat. The Kineto step is skipped.", repeats=5, variant_note="The baseline instrumented build is used throughout. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Dispatch data plane, tail, kernel span</b> from the SM 0 and warp stamps as in E0 (median over ranks and repeats). <b>C1 slot wait</b>, <b>C3 issue</b> and <b>per-lane store latency</b> pooled over all ranks, warps and repeats.</li>
<li><b>Other kernel's time</b>: CUDA events on the side stream around one combine call (alone: before the repeats; overlapped: in each repeat), or around the whole NCCL queue (alone: one call × count; overlapped: the queue).</li>
<li><b>Per-link bytes</b>: NVML <code>NVLINK_THROUGHPUT_DATA_TX/RX</code> per link, after minus before, per repeat, per GPU; the figure shows GPU 4. The spread across links (coefficient of variation) says whether the traffic is balanced over the 18 links.</li>
</ul>
<h3>2.3 Configuration</h3>
<ul><li>4 ranks on host GPUs 4–7, 16 SMs per DeepEP kernel, 4,096 tokens per rank, uniform routing, CPU sync on. NCCL all-to-all through a second process group with 64 MB buffers.</li></ul>

<h2>3. Results</h2>
<div class="finding"><b>A concurrent combine costs little; a concurrent NCCL all-to-all doubles the dispatch.</b> Dispatch data plane: {md(al, 'data'):.0f} µs alone, {md(cb, 'data'):.0f} µs beside combine, {md(n6, 'data'):.0f} µs beside six all-to-alls, {md(n16, 'data'):.0f} µs beside sixteen. The per-lane store latency p50 goes from {P('alone_nonvml', 'lane', 50)/1e3 if al else 0:.1f} to {P('combine_nonvml', 'lane', 50)/1e3 if cb else 0:.1f} / {P('nccl6_nonvml', 'lane', 50)/1e3 if n6 else 0:.1f} / {P('nccl16_nonvml', 'lane', 50)/1e3 if n16 else 0:.1f} µs and the slot wait p99 from {P('alone_nonvml', 'slot', 99)/1e3 if al else 0:.0f} to {P('combine_nonvml', 'slot', 99)/1e3 if cb else 0:.0f} / {P('nccl6_nonvml', 'slot', 99)/1e3 if n6 else 0:.0f} / {P('nccl16_nonvml', 'slot', 99)/1e3 if n16 else 0:.0f} µs. The combine takes {cb['ov_alone']*1e3 if cb else 0:.0f} µs alone and {statistics.median(cb['ov_over'])*1e3 if cb and cb['ov_over'] else 0:.0f} µs beside the dispatch.</div>
{report.figure(fig1, 'Figure 2. Dispatch under sharing: data plane, slot wait and per-lane store latency per case.')}
{report.figure(fig2, 'Figure 3. The other kernel: combine (one call) and the NCCL queue, alone and while the dispatch runs.')}
<div class="finding"><b>The signature is the same as link saturation by the dispatch’s own fan-out.</b> Longer slot waits and longer per-lane waits, uniformly across destinations, are exactly what E0 shows when the SM count is raised past the link’s capacity. Nothing in the dispatch’s stamps says that another kernel is the cause. The per-link counters do: the TX bytes on GPU 4 per repeat rise from the dispatch’s own share to the sum of both kernels (table), and they are spread evenly over the 18 links in every case, so the sharing is on every port, not on a few.</div>
{report.figure(fig3, 'Figure 4. Per-link TX and RX bytes on GPU 4 during one traced repeat, per case (runs with the NVML read). The dispatch alone sends 161 MB from this GPU; the other kernels add theirs on the same links.')}
<div class="finding"><b>The monitor perturbed the measurement.</b> The first set of runs read the NVML per-link counters on rank 0 a few tens of ms before each launch. With that read the dispatch's data plane was {md(al_nv, 'data'):.0f} µs instead of {md(al, 'data'):.0f} µs and the opening barrier showed 100–400 µs of launch stagger between ranks; the effect lasts about a millisecond after the read, so cases whose side-stream launch took longer than that (combine) were not affected. The timing comparisons above therefore use repeated runs without the read, and the counter runs are used only for the per-link byte distribution. For a tool, this is a measurement in itself: an NVML field-value query is not a free observer of a sub-millisecond kernel.</div>
{report.table(['case', 'dispatch span µs', 'data plane µs', 'warp tail µs', 'slot wait p50/p99 µs', 'C3 issue p50 µs', 'lane issue→return p50/p99 µs', 'other kernel alone µs', 'other kernel overlapped µs', 'TX all GPUs MB/repeat', 'per-link CV %'], tab, num_cols=range(1, 11))}

<h2>4. Conclusions</h2>
<ol>
<li><b>Who pays depends on the kernels, and the loser cannot tell.</b> Beside a combine on another buffer, both DeepEP kernels keep their time to within 10 % (the two together stay under the link capacity); beside a NCCL all-to-all the dispatch's data plane doubles while the all-to-all queue slows by about a third. From inside either kernel the slowdown is indistinguishable from the kernel’s own link saturation (longer slot waits and per-lane waits, uniformly over destinations). This is the clearest case for a signal outside the kernel.</li>
<li><b>Observers are not free.</b> An NVML counter read before the launch cost the dispatch about 20 % and hundreds of µs of launch stagger; a tool must either read counters from the GPU or place host reads where their effect has decayed.</li>
<li><b>Link counters give the attribution, at coarse time resolution.</b> Per-link byte deltas over the kernel show whose bytes were there; what they cannot give is <em>when</em> within the kernel, because one NVML read of four GPUs takes milliseconds. A per-link byte counter readable from the GPU, or sampled at tens of µs, would turn this into a per-phase attribution.</li>
<li><b>Dispatch and combine on separate buffers co-exist without deadlock</b> on this build, at the cost measured here; the pipelined-MoE overlap is therefore a scenario the traced kernel can be run in directly.</li>
<li><b>Balanced links.</b> In all cases the 18 links of a GPU carry equal shares (CV of a few percent), so on this node the address-to-link mapping does not concentrate either kernel’s traffic; a link-level hot spot is not among the causes to look for here.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Four ranks; the combine is replayed from one fixed handle; the NCCL traffic pattern is a plain all-to-all; NVML resolution is per repeat, not per phase; the other kernel is not traced, so its own phase decomposition is not available.</div>
'''
meta = f"experiment E8 · {len(allnames)} runs · 4 × H100 on host GPUs 4–7 · scripts: e8_sweep.py, common/run_dispatch.py (--overlap, --nvml), e8_analyze.py · generated {time.strftime('%Y-%m-%d %H:%M')}"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E8 · Two kernels on the same links', 'Dispatch beside a concurrent combine or a NCCL all-to-all: what each pays, what the stamps show, and what only the link counters can attribute.', meta, body))
print('wrote report with', len(names), 'cases')
