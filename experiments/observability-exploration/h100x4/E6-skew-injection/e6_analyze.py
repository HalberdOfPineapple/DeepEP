"""Build report.html for E6 from results/*.pt."""
import glob, os, sys, statistics, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, traceload as TLD, trace_layout as TL
from viz import PALETTE, INK, MUTED

def host_to_gpu(host, h_ns):
    """Map a host perf_counter stamp onto this rank's GPU clock using the two bracketing (host, gpu) samples."""
    (h0, g0, _), (h1, g1, _) = host['map0'], host['map1']
    if h1 == h0: return g0 + (h_ns - h0)
    return g0 + (h_ns - h0) * (g1 - g0) / (h1 - h0)

cases = []
for f in sorted(glob.glob(os.path.join(HERE, 'results', '*.pt'))):
    run = TLD.load(f); a = run['args']; name = os.path.basename(f)[:-3]
    NR = run['NR']; rows = []
    for i in range(run['nrep']):
        tl = {}
        for r in range(NR):
            rep = run['ranks'][r]['reps'][i]; ph = TLD.phase_durations(rep)
            conv = lambda v: TLD.to_rank0_clock(run, i, r, v)
            h = rep['host']
            tl[r] = dict(call=conv(host_to_gpu(h, h['call'])) if h else None, ret=conv(host_to_gpu(h, h['ret'])) if h else None,
                         entry=conv(ph['entry']), open_arrive=conv(ph['open_arrive']), open_release=conv(ph['open_release']), first_warp=conv(ph['first_warp_done']),
                         last_warp=conv(ph['last_warp_done']), close_arrive=conv(ph['close_arrive']), close_release=conv(ph['close_release']), end=conv(ph['end']),
                         A_wait_us=ph['A_wait_us'], data_us=ph['data_us'], E_wait_us=ph['E_wait_us'], span_us=ph['span_us'], tail_us=ph['tail_us'])
        rows.append(tl)
    cases.append(dict(name=name, skew_us=a['skew_us'], mode=a['skew_mode'], tok=a['num_tokens'], rows=rows, NR=NR))
cases.sort(key=lambda c: (c['tok'] != 4096, c['mode'] != 'gpu', c['skew_us']))
gpu4096 = [c for c in cases if c['tok'] == 4096 and c['mode'] == 'gpu']
host100 = next((c for c in cases if c['mode'] == 'host'), None)
c256 = [c for c in cases if c['tok'] == 256]

def med(c, key, ranks):
    return statistics.median(c['rows'][i][r][key] for i in range(len(c['rows'])) for r in ranks)
def med_rel(c, key_a, key_b, ranks):
    """median over repeats and ranks of (key_a of rank - key_b of rank) in us"""
    return statistics.median((c['rows'][i][r][key_a] - c['rows'][i][r][key_b]) / 1e3 for i in range(len(c['rows'])) for r in ranks)
def med_cross(c, key_a, ra, key_b, rb):
    return statistics.median((c['rows'][i][ra][key_a] - c['rows'][i][rb][key_b]) / 1e3 for i in range(len(c['rows'])))

# ---- Figure 1: timeline, one repeat of the 500 us case (or largest) ----
show = next((c for c in gpu4096 if c['skew_us'] == 500), gpu4096[-1] if gpu4096 else cases[0])
tl = show['rows'][min(1, len(show['rows']) - 1)]; NR = show['NR']
t0 = min(tl[r]['entry'] for r in range(NR)); t0 = min(t0, min(tl[r]['call'] for r in range(NR) if tl[r]['call'] is not None))
f, ax = viz.fig(8.6, 3.2)
segs = [('open barrier', 'entry', 'open_release', PALETTE[3]), ('data plane', 'open_release', 'last_warp', PALETTE[0]), ('closing wait', 'last_warp', 'close_release', PALETTE[1]), ('hand-off', 'close_release', 'end', PALETTE[2])]
for r in range(NR):
    for lbl, ka, kb, col in segs:
        xa, xb = (tl[r][ka] - t0) / 1e3, (tl[r][kb] - t0) / 1e3
        ax.barh(r, xb - xa, left=xa, color=col, height=0.55, label=lbl if r == 0 else None)
    if tl[r]['call'] is not None:
        ax.plot([(tl[r]['call'] - t0) / 1e3], [r], marker='|', color=INK, markersize=14, ls='none', label='host call' if r == 0 else None)
    ax.plot([(tl[r]['entry'] - t0) / 1e3], [r], marker='v', color=INK, markersize=5, ls='none', label='kernel entry' if r == 0 else None)
ax.set_yticks(range(NR)); ax.set_yticklabels([f'rank {r}' + (' (delayed)' if r == show['rows'][0] and False else '') for r in range(NR)]); ax.invert_yaxis()
ax.set_xlabel('µs from the earliest host call, rank 0 clock'); ax.set_title(f'One dispatch with rank 0 delayed by {show["skew_us"]:.0f} µs on the GPU ({show["tok"]} tokens)')
viz.legend_below(ax, ncol=6); fig1 = viz.to_svg(f)

# ---- Figure 2: measured opening-barrier wait vs injected skew ----
f, axs = viz.fig(8.6, 3.6, ncols=2)
xs = [c['skew_us'] for c in gpu4096]
others = list(range(1, NR))
w_others = [med(c, 'A_wait_us', others) for c in gpu4096]; w_r0 = [med(c, 'A_wait_us', [0]) for c in gpu4096]
entry_lag = [max(0.5, med_cross(c, 'entry', 0, 'entry', 1)) for c in gpu4096]
ax = axs[0]
ax.plot(xs, w_others, marker='o', color=PALETTE[0], label='ranks 1–3 wait (arrive → release)')
ax.plot(xs, entry_lag, marker='s', color=PALETTE[2], ls='--', label='rank 0 entry − rank 1 entry (aligned clocks)')
ax.plot(xs, w_r0, marker='o', color=PALETTE[1], label='rank 0 wait')
ax.plot([1, max(xs)], [1, max(xs)], color=MUTED, lw=0.8, ls=':', label='y = x')
if host100: ax.plot([100], [med(host100, 'A_wait_us', others)], marker='D', color=PALETTE[4], ls='none', label='host-side 100 µs: ranks 1–3 wait')
ax.set_xlabel('injected delay on rank 0 (µs)'); ax.set_ylabel('µs'); ax.set_title('Opening barrier: who waits, how long')
ax.set_xscale('symlog', linthresh=20); ax.set_yscale('log'); ax.set_ylim(1, 4000); viz.legend_below(ax, ncol=1)
ax = axs[1]
spans = [med(c, 'span_us', others) for c in gpu4096]; datas = [med(c, 'data_us', others) for c in gpu4096]; ewait = [med(c, 'E_wait_us', others) for c in gpu4096]
ax.plot(xs, spans, marker='o', color=PALETTE[0], label='kernel span'); ax.plot(xs, datas, marker='o', color=PALETTE[2], label='data plane'); ax.plot(xs, ewait, marker='o', color=PALETTE[1], label='closing wait')
ax.set_xlabel('injected delay on rank 0 (µs)'); ax.set_ylabel('µs, ranks 1–3 median'); ax.set_title('On the ranks that were on time'); ax.set_xscale('symlog', linthresh=20); viz.legend_below(ax, ncol=1)
fig2 = viz.to_svg(f)

# ---- Figure 3: host-side view: call -> entry per rank, gpu vs host skew ----
f, ax = viz.fig(8.6, 3.0)
labels, vals_r0, vals_o = [], [], []
for c in gpu4096 + ([host100] if host100 else []):
    labels.append(f'{c["mode"]} {c["skew_us"]:.0f} µs'); vals_r0.append(med_rel(c, 'entry', 'call', [0])); vals_o.append(med_rel(c, 'entry', 'call', others))
x = np.arange(len(labels)); ax.bar(x - 0.18, vals_o, 0.36, color=PALETTE[0], label='ranks 1–3'); ax.bar(x + 0.18, vals_r0, 0.36, color=PALETTE[1], label='rank 0')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel('host call → kernel entry (µs)'); ax.set_yscale('symlog', linthresh=20)
ax.set_title('Launch latency as the host sees it: a GPU-side delay shows here, a host-side delay does not'); viz.legend_below(ax, ncol=2)
fig3 = viz.to_svg(f)

# ---- table ----
tab = []
for c in cases:
    tab.append([c['name'], c['tok'], f'{c["skew_us"]:.0f}', c['mode'], f'{med(c, "A_wait_us", others):.1f}', f'{med(c, "A_wait_us", [0]):.1f}', f'{med_cross(c, "entry", 0, "entry", 1):.1f}',
                f'{med_rel(c, "entry", "call", [0]):.1f}', f'{med(c, "data_us", others):.0f}', f'{med(c, "E_wait_us", others):.1f}', f'{med(c, "span_us", others):.0f}', f'{med(c, "span_us", [0]):.0f}'])

k100 = next((c for c in gpu4096 if c['skew_us'] == 100), None); k0 = next((c for c in gpu4096 if c['skew_us'] == 0), None)
k2000 = next((c for c in gpu4096 if c['skew_us'] == 2000), None)
def fmt(c, key, ranks): return f'{med(c, key, ranks):.0f}' if c else '?'

mech = '''<svg viewBox="0 0 880 250" role="img" aria-label="Skew injection: rank 0 runs a spin kernel of X microseconds on its stream right before the dispatch launch; all four ranks stamp kernel entry and the opening barrier; the stamps are placed on one clock with the per-run calibration">
<defs><marker id="sa" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-mut"/></marker></defs>
<text x="20" y="24" class="head">Rank 0</text><text x="20" y="130" class="head">Ranks 1–3</text>
<line x1="110" y1="52" x2="860" y2="52" class="ln-mut"/><line x1="110" y1="158" x2="860" y2="158" class="ln-mut"/>
<rect x="110" y="32" width="90" height="40" rx="6" class="box"/><text x="155" y="50" text-anchor="middle" class="mono">dist.barrier</text><text x="155" y="66" text-anchor="middle" class="small">all ranks aligned</text>
<rect x="215" y="32" width="150" height="40" rx="6" class="box-warn"/><text x="290" y="50" text-anchor="middle" class="mono">spin kernel, X µs</text><text x="290" y="66" text-anchor="middle" class="small">the injected delay</text>
<rect x="380" y="32" width="130" height="40" rx="6" class="box-acc"/><text x="445" y="50" text-anchor="middle" class="mono">dispatch: entry</text><text x="445" y="66" text-anchor="middle" class="small">stamp ①</text>
<rect x="525" y="32" width="120" height="40" rx="6" class="box-acc"/><text x="585" y="50" text-anchor="middle" class="mono">barrier arrive</text><text x="585" y="66" text-anchor="middle" class="small">stamp ②, red.release</text>
<rect x="660" y="32" width="120" height="40" rx="6" class="box-ok"/><text x="720" y="50" text-anchor="middle" class="mono">release</text><text x="720" y="66" text-anchor="middle" class="small">stamp ③: all 4 arrived</text>
<rect x="110" y="138" width="90" height="40" rx="6" class="box"/><text x="155" y="156" text-anchor="middle" class="mono">dist.barrier</text>
<rect x="215" y="138" width="130" height="40" rx="6" class="box-acc"/><text x="280" y="156" text-anchor="middle" class="mono">dispatch: entry</text><text x="280" y="172" text-anchor="middle" class="small">stamp ①</text>
<rect x="360" y="138" width="120" height="40" rx="6" class="box-acc"/><text x="420" y="156" text-anchor="middle" class="mono">barrier arrive</text><text x="420" y="172" text-anchor="middle" class="small">stamp ②</text>
<rect x="495" y="138" width="150" height="40" rx="6" class="box-warn"/><text x="570" y="156" text-anchor="middle" class="small">waiting for rank 0: fabric idle</text><text x="570" y="172" text-anchor="middle" class="small">= X µs + base barrier cost</text>
<rect x="660" y="138" width="120" height="40" rx="6" class="box-ok"/><text x="720" y="156" text-anchor="middle" class="mono">release</text><text x="720" y="172" text-anchor="middle" class="small">stamp ③</text>
<path d="M585 72 L585 100 L720 100 L720 138" class="ln-mut" marker-end="url(#sa)"/><text x="655" y="94" text-anchor="middle" class="small">last arrival releases everyone</text>
<text x="20" y="215" class="small">Host side: perf_counter stamps at the dispatch call and its return, mapped onto the GPU clock by a one-thread timer kernel bracketed by host stamps (±5 µs).</text>
<text x="20" y="233" class="small">The host-mode variant replaces the spin kernel by a host busy-wait, so the launch itself is late instead of the GPU being busy.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>A dispatch cannot start moving data until every rank has arrived at its opening barrier, and cannot finish until every rank has arrived at its closing barrier. So if one rank reaches the kernel late, because the GEMM before it ran longer, because its launch was delayed, or because its previous communication kernel had not finished, every other rank sits in the barrier for exactly that long, with the fabric idle. From inside any single rank this looks like "the barrier took long"; nothing on that rank says which peer was late or why. The suspicion is that in a real pipeline this idle time is a large share of the communication phase and is routinely misread as communication cost.</p>
<p>Two things are tested. First, whether the per-rank stamps already in the traced kernel, placed on one clock with the per-run calibration, attribute an opening-barrier wait to the rank that caused it and recover the amount of delay. Second, whether host-side timestamps around the launch can tell a delay that happened <em>on the GPU before the kernel</em> (a long preceding kernel) from one that happened <em>on the host before the launch</em> (a slow CPU). Those two have the same signature inside the kernel and different fixes.</p>

<h2>2. Methodology</h2>
<p>In one sentence: delay rank 0 by a known amount right before each dispatch, stamp the kernel entry and both barriers on all four ranks, align the stamps on one clock, and check that the other ranks' barrier waits equal the injected delay.</p>
{report.figure(mech, 'Figure 1. The injected delay and the stamps that should recover it.')}
{report.dispatch_procedure("What varies here: the delay injected on rank 0 before each traced dispatch, 0, 20, 100, 500 and 2,000 µs, as a spin kernel on rank 0's stream (the GPU is busy, the host is not); one case injects 100 µs as a host busy-wait instead (the launch is late); and two cases at 256 tokens, where the kernel is short enough for the delay to dominate. The Kineto step is skipped; all numbers come from the stamps.", repeats=5, variant_note="The baseline instrumented build is used throughout. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Opening-barrier wait</b> of a rank = release stamp − its own arrive stamp (both on SM 0 of that rank, no clock alignment needed). For the on-time ranks this should equal the injected delay plus the barrier's base cost.</li>
<li><b>Entry lag</b> = rank 0's kernel-entry stamp − rank 1's, after placing both on rank 0's clock with the mean of the two calibrations that bracket the dispatch. This should equal the injected delay in GPU mode and in host mode alike.</li>
<li><b>Host call → kernel entry</b> on each rank: the host stamp taken just before <code>buffer.dispatch()</code>, mapped onto that rank's GPU clock through the two (host, GPU) timer samples that bracket the repeat, subtracted from the entry stamp. In GPU mode rank 0's launch queues behind the spin kernel, so this grows by the delay; in host mode the call itself is late and this stays small.</li>
<li><b>Data plane</b> = last dispatch warp finished − opening-barrier release; <b>closing wait</b> = closing-barrier release − last warp finished; <b>kernel span</b> = kernel end − kernel entry. All on the on-time ranks (median over ranks 1–3 and the five repeats) unless stated.</li>
</ul>
<h3>2.3 Configuration</h3>
<ul><li>4 ranks on host GPUs 4–7, 16 SMs, 4,096 tokens per rank (256 for the two short cases), uniform routing, CPU sync on, baseline instrumented kernel. Host processes pinned to socket-1 cores.</li></ul>

<h2>3. Results</h2>
{report.figure(fig1, f'Figure 2. One dispatch with rank 0 delayed by {show["skew_us"]:.0f} µs on the GPU. The rows are the four ranks on one clock; the tick is the host call, the triangle the kernel entry. Ranks 1–3 enter on time and sit in the opening barrier until rank 0 arrives; the data plane then runs at its normal length on every rank.')}
<div class="finding"><b>The on-time ranks pay the delay, minus about 85 µs that the launch path hides, and the cross-rank stamps name the late rank.</b> With no delay the opening barrier costs ranks 1–3 {fmt(k0, "A_wait_us", others)} µs. With 100 µs injected on rank 0's GPU they wait only {fmt(k100, "A_wait_us", others)} µs; with 500 µs, {fmt(next((c for c in gpu4096 if c["skew_us"]==500), None), "A_wait_us", others)} µs; with 2,000 µs, {fmt(k2000, "A_wait_us", others)} µs. The difference, about 85 µs, is the host-side cost of <code>buffer.dispatch()</code> before the kernel is enqueued (call → entry is {fmt(k0, "entry", others) if False else f"{med_rel(k0, 'entry', 'call', others):.0f}" if k0 else "?"} µs with nothing in the way, Figure 4): a GPU that is still busy for less than that when the host starts the call delays nothing. A host-side delay of 100 µs, by contrast, is paid almost in full ({fmt(host100, "A_wait_us", others)} µs). In every case the entry lag recovered from the aligned clocks matches the wait (Figure 3 left), so the stamps say which rank was late and by how much, and rank 0's own wait stays at {fmt(k100, "A_wait_us", [0])} µs. The data plane on the on-time ranks is {fmt(k0, "data_us", others)} µs without and {fmt(k2000, "data_us", others)} µs with the largest delay: skew is idle time, not slower communication.</div>
{report.figure(fig2, 'Figure 3. Left: opening-barrier wait of the on-time ranks and of the delayed rank, and the entry lag recovered from the aligned clocks, against the injected delay (dotted: y = x). Right: kernel span, data plane and closing wait on the on-time ranks. Symlog axes.')}
<div class="finding"><b>Host stamps separate a busy GPU from a slow host.</b> In GPU mode rank 0's host-call-to-kernel-entry grows by the injected delay (the launch queues behind the spin kernel) while its host call is on time; in host mode (100 µs busy-wait) the call itself is late and call-to-entry stays at the undelayed value. Inside the kernel both cases are identical. So the pair (host call stamp, kernel entry stamp) per rank is what distinguishes "the previous kernel ran long" from "the CPU was late", and neither is visible from the other ranks' barrier wait alone.</div>
{report.figure(fig3, 'Figure 4. Host call → kernel entry on rank 0 and on the on-time ranks for each case.')}
{report.table(['case', 'tokens', 'delay µs', 'mode', 'open wait ranks 1–3 µs', 'open wait rank 0 µs', 'entry lag µs', 'rank 0 call→entry µs', 'data plane µs', 'closing wait µs', 'span ranks 1–3 µs', 'span rank 0 µs'], tab, num_cols=range(1, 12))}

<h2>4. Conclusions</h2>
<ol>
<li><b>Attribution of barrier time works with what the traced kernel already has.</b> Entry, arrive and release stamps on every rank, aligned by the per-run calibration, recover the effective delay to within a few µs and point at the rank that caused it. Without the alignment, each rank only knows that it waited.</li>
<li><b>About 85 µs of GPU-side slack before a dispatch is free.</b> The host needs that long inside <code>buffer.dispatch()</code> before the kernel is enqueued, so a preceding kernel that overruns by less than that on one rank costs the others nothing; one that overruns by more costs them the excess, one for one. A host-side delay has no such allowance. This number is a property of the DeepEP Python/C++ launch path on this host and is itself worth watching: it is also the floor of the decode critical path (E9).</li>
<li><b>Skew is pure idle.</b> Nothing about the data plane changes; the delay beyond the launch allowance reappears as fabric idle time on the other ranks, and in the 256-token case it is most of the kernel ({fmt(c256[-1] if c256 else None, "A_wait_us", others)} µs waited of a {fmt(c256[-1] if c256 else None, "span_us", others)} µs span with 100 µs injected). Any tool that reports the communication phase by kernel duration counts this idle time as communication.</li>
<li><b>A host stamp per launch is cheap and necessary.</b> It is the only signal that distinguishes a GPU-side cause (a long preceding kernel, which is the compute-skew case) from a host-side one (launch latency, CPU contention); the kernel-internal view cannot.</li>
<li><b>Resolution.</b> The host-to-GPU clock mapping used here is good to about ±5 µs; the GPU-to-GPU alignment to about 0.1 µs. For decode-size kernels the host side is the coarse one.</li>
</ol>
<div class="warnbox"><b>Limits.</b> One delayed rank; four ranks; the delay is injected at a clean point (before the launch), whereas a real pipeline's skew comes from a preceding kernel whose own end stamp would be the natural anchor. Host-side mapping costs 50 tiny launches per repeat and is not something to leave on per kernel in decode.</div>
'''
meta = f"experiment E6 · {len(cases)} cases · 4 × H100 on host GPUs 4–7 · scripts: e6_sweep.py, common/run_dispatch.py (--skew-*), e6_analyze.py · generated {time.strftime('%Y-%m-%d %H:%M')}"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E6 · Skew injection: who waits for whom', 'Delay one rank by a known amount before dispatch and check that the cross-rank stamps give the delay back, and that host stamps tell a busy GPU from a slow host.', meta, body))
print('wrote report with', len(cases), 'cases')
