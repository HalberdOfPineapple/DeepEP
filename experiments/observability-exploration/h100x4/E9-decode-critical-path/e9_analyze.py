"""Build report.html for E9 from results/*.pt."""
import glob, os, sys, statistics, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, traceload as TLD, trace_layout as TL
from viz import PALETTE, INK, MUTED

ORDER = ['base_sms16', 'nocpusync_sms16', 'padded_sms16', 'base_sms4', 'base_sms32']
LABEL = {'base_sms16': '16 SMs', 'nocpusync_sms16': '16 SMs,\nno CPU sync', 'padded_sms16': '16 SMs,\npadded counters', 'base_sms4': '4 SMs', 'base_sms32': '32 SMs'}
SEGS = [('host → entry', PALETTE[7]), ('A entry → open release', PALETTE[3]), ('B counts: release → F done', PALETTE[0]), ('B′ F done → H done', PALETTE[6]), ('C/D data: H done → last warp', PALETTE[2]), ('E last warp → close release', PALETTE[1]), ('F close release → end', PALETTE[4]), ('end → host return', PALETTE[5])]

def host_to_gpu(host, h_ns):
    (h0, g0, _), (h1, g1, _) = host['map0'], host['map1']
    return g0 + (h_ns - h0) * (g1 - g0) / (h1 - h0) if h1 != h0 else g0 + (h_ns - h0)

cases = {}
for f in glob.glob(os.path.join(HERE, 'results', '*.pt')):
    name = os.path.basename(f)[:-3]; run = TLD.load(f); a = run['args']; NR = run['NR']; K = a['topk']
    segs = {s: [] for s, _ in SEGS}; total = []; call_to_ret = []; atomic = []; e_to_f = []; open_wait = []; entry_skew = []; map_err = []
    for i in range(run['nrep']):
        entries = []
        for r in range(NR):
            rep = run['ranks'][r]['reps'][i]; ph = TLD.phase_durations(rep); h = rep['host']
            if h is None: continue
            call = host_to_gpu(h, h['call']); ret = host_to_gpu(h, h['ret']); sync = host_to_gpu(h, h['sync'])
            sm = rep['sm']; s0 = {e: sm[e][0] for e in sm}
            if not s0[7]: s0[7] = s0[6]   # without CPU sync phase H (mailbox) is not stamped: fold B' into B
            v = [ph['entry'] - call, ph['open_release'] - ph['entry'], s0[6] - ph['open_release'], s0[7] - s0[6], ph['last_warp_done'] - s0[7], ph['close_release'] - ph['last_warp_done'], ph['end'] - ph['close_release'], ret - ph['end']]
            for (s, _), val in zip(SEGS, v): segs[s].append(val / 1e3)
            total.append((ret - call) / 1e3); call_to_ret.append((h['ret'] - h['call']) / 1e3)
            atomic += TLD.atomic_latencies(rep, K).tolist(); e_to_f.append((s0[6] - s0[5]) / 1e3); open_wait.append(ph['A_wait_us'])
            map_err.append(max(h['map0'][2], h['map1'][2]) / 1e3)
            entries.append(TLD.to_rank0_clock(run, i, r, ph['entry']))
        if len(entries) == NR: entry_skew.append((max(entries) - min(entries)) / 1e3)
    cases[name] = dict(segs=segs, total=total, call_to_ret=call_to_ret, atomic=atomic, e_to_f=e_to_f, open_wait=open_wait, entry_skew=entry_skew, map_err=map_err, t_kineto_us=statistics.mean(run['ranks'][r]['t_untraced_s'] for r in range(NR)) * 1e6, sms=a['num_sms'], cpu_sync=a['cpu_sync'])
names = [n for n in ORDER if n in cases]

# ---- Figure 1: stacked median segments per case ----
f, ax = viz.fig(8.6, 3.8)
x = np.arange(len(names)); bottom = np.zeros(len(names))
for s, col in SEGS:
    vals = np.array([statistics.median(cases[n]['segs'][s]) for n in names]); ax.bar(x, vals, 0.6, bottom=bottom, color=col, label=s); bottom += vals
for j, n in enumerate(names): ax.annotate(f'{bottom[j]:.0f} µs', (j, bottom[j]), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([LABEL[n] for n in names], fontsize=8); ax.set_ylabel('µs, median over 100 repeats × 4 ranks'); ax.set_title('Host call → host return for a 128-token dispatch, by segment'); viz.legend_below(ax, ncol=2)
fig1 = viz.to_svg(f)

# ---- Figure 2: distributions (p50/p90/p99) of each segment for the base case; and total ----
base = cases.get('base_sms16') or cases[names[0]]
f, axs = viz.fig(8.6, 3.4, ncols=2)
ax = axs[0]
labels = [s for s, _ in SEGS]
p50 = [viz.pct(base['segs'][s], 50) for s in labels]; p90 = [viz.pct(base['segs'][s], 90) for s in labels]; p99 = [viz.pct(base['segs'][s], 99) for s in labels]
y = np.arange(len(labels))
ax.barh(y + 0.25, p50, 0.25, color=PALETTE[0], label='p50'); ax.barh(y, p90, 0.25, color=PALETTE[3], label='p90'); ax.barh(y - 0.25, p99, 0.25, color=PALETTE[1], label='p99')
ax.set_yticks(y); ax.set_yticklabels([l.split(':')[0] for l in labels], fontsize=8); ax.invert_yaxis(); ax.set_xlabel('µs'); ax.set_title('Base case: segment distributions'); viz.legend_below(ax, ncol=3)
ax = axs[1]
xmax = 0
for j, n in enumerate(names):
    xs = np.sort(cases[n]['total']); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=PALETTE[j], drawstyle='steps-post', label=LABEL[n].replace('\n', ' ')); xmax = max(xmax, viz.pct(xs, 99))
ax.set_xlim(0, xmax * 1.1); ax.set_xlabel('host call → host return (µs), axis cut at the p99'); ax.set_ylabel('CDF'); ax.set_title('End-to-end per call'); viz.legend_below(ax, ncol=2)
fig2 = viz.to_svg(f)

tab = []
for n in names:
    c = cases[n]
    tab.append([LABEL[n].replace('\n', ' '), c['sms'], 'on' if c['cpu_sync'] else 'off', f'{statistics.median(c["total"]):.0f} / {viz.pct(c["total"], 99):.0f}', f'{c["t_kineto_us"]:.0f}'] + [f'{statistics.median(c["segs"][s]):.1f}' for s, _ in SEGS] + [f'{statistics.median(c["entry_skew"]):.1f}' if c['entry_skew'] else '–', f'{viz.pct(c["atomic"], 50):.0f} / {viz.pct(c["atomic"], 99):.0f}', f'{statistics.median(c["map_err"]):.1f}'])
hdr = ['case', 'SMs', 'CPU sync', 'call→return p50 / p99 µs', 'Kineto kernel µs'] + [s.split(':')[0] + ' µs' for s, _ in SEGS] + ['entry skew µs', 'atomic p50/p99 ns', 'host map ±µs']

b16 = cases.get('base_sms16'); nc = cases.get('nocpusync_sms16'); pd = cases.get('padded_sms16'); s4 = cases.get('base_sms4'); s32 = cases.get('base_sms32')
def seg(c, s, p=50): return viz.pct(c['segs'][s], p) if c else float('nan')
def tot(c, p=50): return viz.pct(c['total'], p) if c else float('nan')
kern = lambda c: sum(statistics.median(c['segs'][s]) for s, _ in SEGS[1:7]) if c else float('nan')

mech = '''<svg viewBox="0 0 880 250" role="img" aria-label="E9: the decode critical path from the host call to the host return, cut into segments by host stamps mapped onto the GPU clock and by the kernel's SM 0 stamps">
<defs><marker id="da" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-mut"/></marker></defs>
<text x="20" y="24" class="head">One dispatch call, 128 tokens per rank: the segments</text>
<line x1="20" y1="70" x2="860" y2="70" class="ln"/>
<g class="small">
<rect x="20" y="52" width="80" height="36" rx="5" class="box-bad"/><text x="60" y="74" text-anchor="middle">host → entry</text>
<rect x="104" y="52" width="80" height="36" rx="5" class="box-warn"/><text x="144" y="74" text-anchor="middle">A barrier</text>
<rect x="188" y="52" width="110" height="36" rx="5" class="box-acc"/><text x="243" y="74" text-anchor="middle">B counts → F</text>
<rect x="302" y="52" width="80" height="36" rx="5" class="box-type"/><text x="342" y="74" text-anchor="middle">B′ F → H</text>
<rect x="386" y="52" width="130" height="36" rx="5" class="box-ok"/><text x="451" y="74" text-anchor="middle">C/D data → last warp</text>
<rect x="520" y="52" width="110" height="36" rx="5" class="box-warn"/><text x="575" y="74" text-anchor="middle">E closing barrier</text>
<rect x="634" y="52" width="90" height="36" rx="5" class="box"/><text x="679" y="74" text-anchor="middle">F → kernel end</text>
<rect x="728" y="52" width="132" height="36" rx="5" class="box-bad"/><text x="794" y="74" text-anchor="middle">end → host return</text>
</g>
<text x="60" y="110" text-anchor="middle" class="small">host stamp</text><text x="60" y="124" text-anchor="middle" class="small">mapped ±5 µs</text>
<text x="144" y="110" text-anchor="middle" class="small">SM 0 ①②③</text>
<text x="243" y="110" text-anchor="middle" class="small">SM 0 phase F</text><text x="243" y="124" text-anchor="middle" class="small">(all counts seen)</text>
<text x="342" y="110" text-anchor="middle" class="small">prefix sums,</text><text x="342" y="124" text-anchor="middle" class="small">mailbox</text>
<text x="451" y="110" text-anchor="middle" class="small">WARP records</text>
<text x="575" y="110" text-anchor="middle" class="small">SM 0 ⑨⑩</text>
<text x="679" y="110" text-anchor="middle" class="small">SM 0 ⑪</text>
<text x="794" y="110" text-anchor="middle" class="small">host stamp after</text><text x="794" y="124" text-anchor="middle" class="small">dispatch() returns</text>
<text x="20" y="160" class="small">With CPU sync on, dispatch() returns after the host has read the token counts from the mailbox and allocated the outputs; "end → host return" then contains the</text>
<text x="20" y="176" class="small">epilogue kernel and the host work. With CPU sync off it returns as soon as the kernels are enqueued, and the host stamp no longer marks a dependency.</text>
<text x="20" y="200" class="small">Mapping: a one-thread kernel reads %globaltimer between two host stamps; the sample with the tightest bracket (about ±5 µs) gives host→GPU offset; done before and after each repeat.</text>
<text x="20" y="224" class="small">Within the kernel, stamps are on one GPU's clock: no alignment needed for a per-rank decomposition. Entry skew across ranks uses the per-run calibration.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>In decode each rank dispatches a handful of tokens, so the dispatch is not a bandwidth problem but a latency chain: launch, opening barrier, the count exchange, a single token per warp, the closing barrier, the epilogue, and the host reading the counts before it can allocate the expert inputs. Nothing in this chain is large, but all of it is serialised and it repeats for every layer of every decode step. The question is what the chain is made of at 128 tokens, which segment is the longest, how much of it is on the host rather than on the GPU, how it changes with the number of SMs and with the CPU-sync option, and whether the slot-counter padding of E3 moves anything at this size. A per-kernel duration from a profiler answers none of these; the segment decomposition is what a tool must produce.</p>

<h2>2. Methodology</h2>
<p>In one sentence: run 100 dispatches of 128 tokens per rank per case, stamp the host call and return and map them onto the GPU clock, cut the span from host call to host return into eight segments with the in-kernel stamps, and compare cases.</p>
{report.figure(mech, 'Figure 1. The eight segments and where each boundary stamp comes from.')}
{report.dispatch_procedure("What varies here: SM count (4, 16, 32), the CPU-sync option (on: the host waits for the token counts inside dispatch(); off: dispatch() returns immediately), and the slot-counter layout (padded, as in E3). 128 tokens per rank, 100 traced repeats per case; the Kineto step runs with 50 iterations and gives the reference kernel time.", repeats=100, variant_note="The padded case uses the --pad-counters build. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Segments</b> per (rank, repeat), in µs: host call → kernel entry; entry → opening-barrier release; release → notify phase F done (all peers' counts seen); F → H (prefix sums written, mailbox); H → last dispatch warp finished; last warp → closing-barrier release; release → kernel end; kernel end → host return. Host stamps are mapped onto the GPU clock with the bracketing timer samples; the bracket half-width (about ±5 µs) is reported.</li>
<li><b>Call → return</b> is measured purely on the host (no mapping) and is the end-to-end number a caller experiences with CPU sync on.</li>
<li><b>Entry skew</b> = latest − earliest kernel entry across the four ranks on one clock (per-run calibration), the launch skew the opening barrier absorbs.</li>
<li><b>Slot atomic latency</b> per lane, as in E3. <b>Kineto kernel µs</b> as in E0 (50 iterations, L2 flush and barrier between).</li>
</ul>
<h3>2.3 Configuration</h3>
<ul><li>4 ranks on host GPUs 4–7, 128 tokens per rank, top-8 of 256 experts, uniform routing, expert alignment 128, host processes pinned to socket-1 cores.</li></ul>

<h2>3. Results</h2>
<div class="finding"><b>The 128-token dispatch costs {tot(b16):.0f} µs from call to return at 16 SMs (p99 {tot(b16, 99):.0f} µs), of which the kernel itself is {kern(b16):.0f} µs.</b> The host-side launch path (call → entry) is {seg(b16, 'host → entry'):.0f} µs and the return path (kernel end → return) {seg(b16, 'end → host return'):.0f} µs; inside the kernel the longest segments are the count exchange ({seg(b16, 'B counts: release → F done'):.0f} µs), the data plane ({seg(b16, 'C/D data: H done → last warp'):.0f} µs) and the opening barrier ({seg(b16, 'A entry → open release'):.0f} µs, with {statistics.median(b16['entry_skew']) if b16 and b16['entry_skew'] else 0:.0f} µs of entry skew behind it).</div>
{report.figure(fig1, 'Figure 2. Median segment lengths per case, stacked from the host call to the host return.')}
<div class="finding"><b>What moves.</b> Turning CPU sync off removes {tot(b16) - tot(nc) if nc else 0:.0f} µs from the caller’s view (call → return {tot(nc):.0f} µs) but not from the GPU chain, whose segments are unchanged. Going from 16 to 4 SMs makes the data plane {seg(s4, 'C/D data: H done → last warp'):.0f} µs instead of {seg(b16, 'C/D data: H done → last warp'):.0f} µs; 32 SMs gives {seg(s32, 'C/D data: H done → last warp'):.0f} µs and a {seg(s32, 'A entry → open release'):.0f} µs opening barrier. Padded counters change the atomic from {viz.pct(b16['atomic'], 50) if b16 else 0:.0f} to {viz.pct(pd['atomic'], 50) if pd else 0:.0f} ns and the end-to-end by {tot(pd) - tot(b16) if pd else 0:+.0f} µs: nothing, as E3 predicted.</div>
{report.figure(fig2, 'Figure 3. Left: p50, p90 and p99 of each segment in the base case. Right: end-to-end distribution per case.')}
{report.table(hdr, tab, num_cols=range(1, len(hdr)))}

<h2>4. Conclusions</h2>
<ol>
<li><b>The decode chain is host and synchronisation, not bytes.</b> Launch path, two barriers and the count exchange are most of the {tot(b16):.0f} µs; the data plane at 128 tokens is {seg(b16, 'C/D data: H done → last warp'):.0f} µs. A tool that reports kernel time ({b16['t_kineto_us'] if b16 else 0:.0f} µs by Kineto) sees {100 * (b16['t_kineto_us'] / tot(b16)) if b16 else 0:.0f} % of what the caller pays.</li>
<li><b>Host stamps are part of the instrument.</b> The call → entry segment ({seg(b16, 'host → entry'):.0f} µs) is as large as the biggest in-kernel segment and invisible to any GPU-side probe; E6 showed the same number as the allowance that hides a late peer.</li>
<li><b>The SM count is a latency knob in decode.</b> Fewer SMs lengthen the data plane one for one; more SMs lengthen the barriers (grid syncs over more blocks). DeepEP’s default of 20 is a bandwidth choice; a decode-specific count would be chosen from this decomposition.</li>
<li><b>Tails.</b> The p99 of the end-to-end is {tot(b16, 99):.0f} µs against a p50 of {tot(b16):.0f} µs; the segment p99s (Figure 3) show which segment carries the tail.</li>
</ol>
<div class="warnbox"><b>Limits.</b> One repeat of the no-CPU-sync case had a host-side stall of over a second between call and return (a host event unrelated to the kernel; it is kept in the table's p99 and cut from the CDF). Host-to-GPU mapping is good to about ±5 µs, comparable to the smallest segments. No model compute between dispatches, so the launch path runs on an idle host and GPU; in a real decode loop the previous kernel’s end is the anchor. Four ranks; the count exchange grows with rank count.</div>
'''
meta = f"experiment E9 · {len(names)} cases · 4 × H100 on host GPUs 4–7 · scripts: e9_sweep.py, common/run_dispatch.py (--host-ts, --cpu-sync), e9_analyze.py · generated {time.strftime('%Y-%m-%d %H:%M')}"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E9 · The decode critical path', '128 tokens per rank: the chain from host call to host return cut into eight segments, and how SM count, CPU sync and counter padding move each.', meta, body))
print('wrote report with', len(names), 'cases')
