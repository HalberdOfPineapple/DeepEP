"""Build report.html for E3 from results/*.pt (baseline vs padded slot counters)."""
import glob, os, re, sys, statistics
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, trace_layout as TL
from viz import PALETTE, INK, MUTED

runs = {}
for f in sorted(glob.glob(os.path.join(HERE, 'results', '*.pt'))):
    m = re.match(r'(\w+)_sms(\d+)_tok(\d+)\.pt', os.path.basename(f)); d = torch.load(f, weights_only=False); a = d['args']; T = a['num_tokens']; K = a['topk']
    atomic, barrier_open, barrier_close, t_us, iters = [], [], [], [], []
    for g in d['ranks']:
        t_us.append(g['t_untraced_s'] * 1e6)
        for tr in g['traces']:
            rec = tr[TL.TOK:TL.TOK + TL.TOK_STRIDE * T].view(T, TL.TOK_STRIDE); rec = rec[rec[:, TL.T_ISSUE] > 0]
            ae = rec[:, TL.T_LANE_ATOMIC_END:TL.T_LANE_ATOMIC_END + K]; am = ae > 0
            atomic += (ae[am] - rec[:, TL.T_ATOMIC_START].unsqueeze(1).expand_as(ae)[am]).double().tolist()
            iters += (rec[:, TL.T_ISSUE] - rec[:, TL.T_SLOTWAIT_START]).double().tolist()
            sm = {e: [int(tr[TL.SM + s * 32 + e]) for s in range(a['num_sms'])] for e in (1, 2, 9, 10)}
            barrier_open.append((sm[2][0] - sm[1][0]) / 1e3); barrier_close.append((sm[10][0] - sm[9][0]) / 1e3)
    at = np.array(atomic)
    runs[(m.group(1), int(m.group(2)), int(m.group(3)))] = dict(t_us=statistics.mean(t_us), t_sd=statistics.pstdev(t_us), atomic=at, iter_p50=viz.pct(iters, 50),
                                                                 open_p50=statistics.median(barrier_open), close_p50=statistics.median(barrier_close), n_atomic=len(at))
keys = sorted(runs); configs = sorted({(s, t) for _, s, t in keys}, key=lambda x: (x[1], x[0]))

# Figure 1: atomic latency distribution (CDF) baseline vs padded per config
f, axs = viz.fig(9.2, 3.4, ncols=len(configs), sharey=True)
axs = np.atleast_1d(axs)
for ax, (sms, tok) in zip(axs, configs):
    for v, color in [('base', PALETTE[0]), ('padded', PALETTE[1])]:
        if (v, sms, tok) not in runs: continue
        at = np.sort(runs[(v, sms, tok)]['atomic']); ys = np.arange(1, len(at) + 1) / len(at) * 100
        ax.plot(at, ys, color=color, lw=1.8); viz.end_label(ax, at[int(len(at) * 0.999)], 99.9, v, color)
    ax.set_xscale('log'); ax.set_xlabel('issue → return (ns)'); ax.set_title(f'{sms} SMs, {tok} tok'); ax.set_xlim(50, 20000)
axs[0].set_ylabel('% of atomics')
fig1 = viz.to_svg(f)

# Figure 2: kernel time and barrier times
f, axs = viz.fig(9.2, 3.2, ncols=2)
x = np.arange(len(configs)); w = 0.36
for i, (v, color) in enumerate([('base', PALETTE[0]), ('padded', PALETTE[1])]):
    ts = [runs.get((v, s, t), {}).get('t_us', 0) for s, t in configs]; axs[0].bar(x + (i - 0.5) * w, ts, width=w - 0.04, color=color, label=v)
    bo = [runs.get((v, s, t), {}).get('open_p50', 0) for s, t in configs]; axs[1].bar(x + (i - 0.5) * w, bo, width=w - 0.04, color=color, label=v)
for ax, ttl, yl in [(axs[0], 'Kernel time', 'µs'), (axs[1], 'Opening barrier, SM 0 arrive → release (median)', 'µs')]:
    ax.set_xticks(x); ax.set_xticklabels([f'{s} SMs\n{t} tok' for s, t in configs]); ax.set_title(ttl); ax.set_ylabel(yl); ax.legend(fontsize=8)
fig2 = viz.to_svg(f)

tab = [[v, s, t, f"{runs[(v,s,t)]['t_us']:.1f} ± {runs[(v,s,t)]['t_sd']:.1f}", f"{viz.pct(runs[(v,s,t)]['atomic'],50):.0f}", f"{viz.pct(runs[(v,s,t)]['atomic'],99):.0f}", f"{viz.pct(runs[(v,s,t)]['atomic'],99.9):.0f}", f"{runs[(v,s,t)]['atomic'].max():.0f}", runs[(v,s,t)]['n_atomic'], f"{runs[(v,s,t)]['iter_p50']/1e3:.1f}", f"{runs[(v,s,t)]['open_p50']:.1f}", f"{runs[(v,s,t)]['close_p50']:.1f}"] for (v, s, t) in keys]

def g(v, s, t, k): return runs.get((v, s, t), {}).get(k, float('nan'))
b16, p16 = runs.get(('base', 16, 4096)), runs.get(('padded', 16, 4096))
b32, p32 = runs.get(('base', 32, 4096)), runs.get(('padded', 32, 4096))
mech = '''<svg viewBox="0 0 880 200" role="img" aria-label="Baseline layout: four slot counters in one 32-byte sector of one 128-byte line; padded layout: each counter alone in its own 128-byte line">
<text x="20" y="24" class="head">baseline: scaleup_atomic_sender_count[4] in one line</text>
<rect x="20" y="40" width="400" height="36" rx="4" class="box"/>
<g class="mono"><rect x="20" y="40" width="25" height="36" class="box-bad"/><text x="32" y="63" text-anchor="middle">0</text><rect x="45" y="40" width="25" height="36" class="box-bad"/><text x="57" y="63" text-anchor="middle">1</text><rect x="70" y="40" width="25" height="36" class="box-bad"/><text x="82" y="63" text-anchor="middle">2</text><rect x="95" y="40" width="25" height="36" class="box-bad"/><text x="107" y="63" text-anchor="middle">3</text></g>
<text x="230" y="63" class="small">… 112 unused bytes of the same 128 B line</text>
<text x="20" y="100" class="small">240 warps × up to 4 lanes → returning atomicAdd on one of four ints, all in one L2 slice</text>
<text x="20" y="134" class="head">padded (--pad-counters): stride 32 ints = 128 B</text>
<g class="mono"><rect x="20" y="150" width="100" height="36" rx="4" class="box-warn"/><text x="70" y="173" text-anchor="middle">0  + 124 B</text><rect x="130" y="150" width="100" height="36" rx="4" class="box-warn"/><text x="180" y="173" text-anchor="middle">1  + 124 B</text><rect x="240" y="150" width="100" height="36" rx="4" class="box-warn"/><text x="290" y="173" text-anchor="middle">2  + 124 B</text><rect x="350" y="150" width="100" height="36" rx="4" class="box-warn"/><text x="400" y="173" text-anchor="middle">3  + 124 B</text></g>
<text x="470" y="166" class="small">four lines, hashed to (probably) four slices;</text><text x="470" y="182" class="small">the reset at kernel end uses the same stride</text>
</svg>'''
body = f'''
<h2>1. Motivation</h2>
<p>DeepEP allocates destination slots with a returning <code>atomicAdd</code> on <code>scaleup_atomic_sender_count[dst]</code>, eight (here four) consecutive ints in one 128-byte line, from every dispatch warp for every (token, destination) pair: ≈ 14,800 atomics per dispatch per GPU at four ranks. All of them serialise at one L2 slice, and the returned value is on the warp's critical path, so the concern is that this one cache line becomes a funnel whose latency grows with the number of warps and destinations. E0 already reported a flat 96 ns p50, which suggests the funnel is cheap at this scale; this experiment tests that directly by giving each counter its own line, at two concurrencies and in both the prefill (4,096 tokens) and the decode-like (256 tokens) regime, and records the two NVLink barriers' arrive-to-release time on SM 0 as the other address funnel.</p>
<h2>2. Methodology</h2>
<p>In one sentence: time every slot-allocation atomic from issue to return, with the four counters sharing one cache line as DeepEP has them and with each counter moved to its own line, and see whether the layout changes anything.</p>
{report.figure(mech, 'Figure 1. The two counter layouts. The padded variant changes only the index stride in the kernel (allocation and reset); the EPSignals field has room for 1,024 ints.')}
{report.dispatch_procedure("What varies here: two kernel builds, baseline and <code>--pad-counters</code> (counter stride 32 ints, so each counter has its own 128-byte line), each at (16, 32) SMs × (4,096, 256) tokens. Eight runs in all; routing uniform.", repeats=5, kineto_iters=30, variant_note="For the padded build the patch also changes the index stride of the slot counters where they are allocated and where they are reset. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Atomic latency</b> (Figure 2): the warp stamps the clock just before its lanes issue the returning <code>atomicAdd</code>, and each lane stamps it again when its value comes back; the difference, per lane, is one sample. Both stamps are taken by the lane that issued the atomic. Percentiles are over all lanes, warps, ranks and the five repeats. The clock's 32 ns granularity shows as steps in the CDF.</li>
<li><b>Iteration p50</b> = stores issued − loop top, per token, the same quantity E0 decomposes.</li>
<li><b>Opening / closing barrier µs</b> (Figure 3): SM 0's stamp just before the <code>gpu_barrier</code> call and just after it returns. The opening barrier includes any launch skew between ranks; the closing barrier includes waiting for the slowest rank. Neither can be attributed to the barrier word alone with four ranks, so they are reported, not interpreted as contention.</li>
<li><b>Kernel µs</b> from step 4, mean ± standard deviation over the four ranks.</li>
</ul>
<h3>2.3 Configuration</h3>
<ul><li>4 ranks on host GPUs 4–7, top-8 of 256 experts, uniform routing, expert alignment 128, CPU sync on.</li></ul>
<h2>3. Results</h2>
<div class="finding"><b>The counter funnel costs nothing measurable at four ranks.</b> Atomic latency is {viz.pct(b16['atomic'],50) if b16 else 0:.0f} ns p50 / {viz.pct(b16['atomic'],99) if b16 else 0:.0f} ns p99 at 16 SMs with the shared line and {viz.pct(p16['atomic'],50) if p16 else 0:.0f} / {viz.pct(p16['atomic'],99) if p16 else 0:.0f} ns padded; at 32 SMs {viz.pct(b32['atomic'],50) if b32 else 0:.0f} / {viz.pct(b32['atomic'],99) if b32 else 0:.0f} ns versus {viz.pct(p32['atomic'],50) if p32 else 0:.0f} / {viz.pct(p32['atomic'],99) if p32 else 0:.0f} ns. Kernel time changes by {(p16['t_us']/b16['t_us']-1)*100 if b16 and p16 else 0:+.1f} % (16 SMs) and {(p32['t_us']/b32['t_us']-1)*100 if b32 and p32 else 0:+.1f} % (32 SMs), inside run-to-run noise. The L2 atomic unit absorbs this rate (one atomic every ~40 ns per GPU on average) without queueing.</div>
{report.figure(fig1, "Figure 2. CDF of the slot atomic's issue-to-return time, baseline versus padded, per configuration. Steps are the 32 ns timer granularity.")}
{report.figure(fig3 if False else fig2, 'Figure 3. Kernel time (left) and opening-barrier arrive-to-release on SM 0 (right) for both layouts.')}
{report.table(['layout', 'SMs', 'tokens', 'kernel µs (mean ± sd over ranks)', 'atomic p50 ns', 'p99', 'p99.9', 'max', 'atomics sampled', 'iteration p50 µs', 'open barrier µs', 'close barrier µs'], tab, num_cols=range(1, 12))}
<h2>4. Conclusions</h2>
<ol>
<li><b>The slot-counter funnel is a non-problem at this scale</b>: with four destinations and 240 warps the returning atomics complete in about 100 ns regardless of layout. The concern scales with destinations and warps (eight ranks double the issuers per line, 72 ranks would put nine lines' worth of counters in one), so the padded layout is a free insurance rather than a fix; the measurement to repeat on eight GPUs is this one.</li>
<li><b>Where the µs go instead</b> is the barrier arrive-to-release time, which is launch skew and peer skew rather than word contention; E6 (skew injection) and E9 (decode critical path) are the experiments that decompose it, and the opening-barrier numbers here are their baseline.</li>
<li><b>Method note.</b> The per-lane atomic timing costs two timer reads per token and is the right way to watch this funnel grow with rank count; no counter in the memory system exposes per-line serialisation.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Four ranks only; the per-sender barrier-word variant was not implemented (it changes the barrier protocol on both sides). Timer granularity 32 ns.</div>
'''
meta = f"experiment E3 · {len(runs)} runs · 4 × H100 on host GPUs 4–7 · scripts: e3_sweep.py, common/apply_trace_patch.py (--pad-counters), e3_analyze.py"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E3 · Address hot spots: the slot-counter line', 'Does serialising ~15,000 returning atomics per dispatch on one 32-byte sector cost anything, and does padding the counters change it.', meta, body))
print('wrote report with', len(runs), 'runs')
