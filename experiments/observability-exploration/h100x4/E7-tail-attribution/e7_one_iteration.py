"""One dispatch, one warp, one iteration: locate the problematic phase and the exact straggler. Writes one-iteration.html."""
import os, sys, time, statistics
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, traceload as TLD, trace_layout as TL
from viz import PALETTE, INK, MUTED
import matplotlib.pyplot as plt

run = TLD.load(os.path.join(HERE, 'results', 'ratio1_smid.pt')); K = run['args']['topk']; NW = 15 * run['sms']
R, REP = 1, 0
rep = run['ranks'][R]['reps'][REP]; ph = TLD.phase_durations(rep); t0 = ph['open_release']
rec = rep['tok']; w = rep['warp']
blk = (rec[:, TL.T_ID] >> 32).numpy(); wid = (rec[:, TL.T_ID] & 0xffffffff).numpy(); gw = blk * 15 + wid
smid = (w[:, 3] >> 32).numpy(); fin = (w[:, 1].numpy() - t0) / 1e3; order = np.argsort(fin)

def parts(g):
    m = gw == g; X = rec[m]; X = X[np.argsort(X[:, TL.T_SLOTWAIT_START].numpy())]
    c1 = (X[:, TL.T_SLOTWAIT_END] - X[:, TL.T_SLOTWAIT_START]).numpy() / 1e3
    c2a = (X[:, TL.T_ATOMIC_START] - X[:, TL.T_SLOTWAIT_END]).numpy() / 1e3
    c2b = (X[:, TL.T_LOAD_DONE] - X[:, TL.T_ATOMIC_START]).numpy() / 1e3
    c3 = (X[:, TL.T_ISSUE] - X[:, TL.T_LOAD_DONE]).numpy() / 1e3
    start = (X[:, TL.T_SLOTWAIT_START].numpy() - t0) / 1e3
    nst = int((X[:, TL.T_LANE_ISSUE:TL.T_LANE_ISSUE + K] > 0).sum())
    return dict(c1=c1, c2a=c2a, c2b=c2b, c3=c3, start=start, stores=nst, n=len(c1))
P = {g: parts(g) for g in range(NW)}
strag, med, fast = int(order[-1]), int(order[len(order) // 2]), int(order[0])
sched = np.array([g % 15 % 4 for g in range(NW)])  # dispatch warp d sits on warp scheduler (d + 4) % 4 = d % 4

# ---- Figure 1: per-iteration stacked phases for the three warps ----
f, axs = viz.fig(8.6, 3.6, ncols=3, sharey=True)
cols = [('c1', 'C1 slot wait', PALETTE[3]), ('c2a', 'C2a slot free → atomic', PALETTE[0]), ('c2b', 'C2b atomic → token loaded', PALETTE[6]), ('c3', 'C3 issue stores', PALETTE[1])]
for ax, (g, lbl) in zip(axs, [(strag, 'straggler'), (med, 'median warp'), (fast, 'fastest warp')]):
    p = P[g]; x = np.arange(1, p['n'] + 1); bottom = np.zeros(p['n'])
    for key, name, col in cols:
        ax.bar(x, p[key], 0.8, bottom=bottom, color=col, label=name); bottom += p[key]
    ax.set_title(f'{lbl}\nblk {g // 15} w{g % 15} SM{smid[g]}, {fin[g]:.0f} µs', fontsize=9); ax.set_xlabel('iteration')
axs[0].set_ylabel('µs in the iteration'); viz.legend_below(axs[0], ncol=2)
fig1 = viz.to_svg(f)

# ---- Figure 2: heat map warps (sorted by finish) x iteration, C3 ----
maxn = max(p['n'] for p in P.values())
M = np.full((NW, maxn), np.nan)
for i, g in enumerate(order): M[i, :P[g]['n']] = P[g]['c3']
f, ax = viz.fig(8.6, 4.0)
im = ax.imshow(M, aspect='auto', cmap='magma_r', vmin=0, vmax=40, interpolation='nearest')
ax.set_xlabel('iteration'); ax.set_ylabel('warp, sorted by finish time (fastest at top)'); ax.set_title('C3 issue time per iteration for all 240 warps (µs)')
cb = f.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('µs')
ax.grid(False)
fig2 = viz.to_svg(f)

# ---- Figure 3: finish by block and by warp scheduler ----
f, axs = viz.fig(8.6, 3.4, ncols=2)
ax = axs[0]
for s in range(4):
    gs = [g for g in range(NW) if sched[g] == s]
    ax.scatter([g // 15 + (s - 1.5) * 0.15 for g in gs], [fin[g] for g in gs], s=12, color=PALETTE[s], label=f'scheduler {s}' + (' (3 dispatch warps)' if s == 3 else ' (4)'))
ax.set_xlabel('block'); ax.set_ylabel('warp finish (µs)'); ax.set_title('Finish by block and by warp scheduler'); viz.legend_below(ax, ncol=2)
ax = axs[1]
blk_med = np.array([np.median(fin[b * 15:(b + 1) * 15]) for b in range(run['sms'])])
dev = np.array([fin[g] - blk_med[g // 15] for g in range(NW)])
for s in range(4):
    ax.scatter(np.full(((sched == s).sum(),), s) + np.random.default_rng(0).uniform(-0.2, 0.2, (sched == s).sum()), dev[sched == s], s=10, color=PALETTE[s])
ax.set_xticks(range(4)); ax.set_xticklabels(['sched 0', 'sched 1', 'sched 2', 'sched 3']); ax.set_ylabel('finish − block median (µs)'); ax.set_title('Within a block: the scheduler decides')
fig3 = viz.to_svg(f)

# numbers
sp, mp, fp = P[strag], P[med], P[fast]
first6 = lambda p: float(p['c3'][:6].sum()); rest = lambda p: float(p['c3'][6:].sum())
c3_all_first = np.median([P[g]['c3'][:6].sum() for g in range(NW)]); c3_all_rest = np.median([P[g]['c3'][6:].sum() for g in range(NW)])
dev3 = float(np.median(dev[sched == 3])); dev012 = float(np.median(dev[sched != 3]))
corr = {k: float(np.corrcoef(fin, [np.sum(P[g][k]) for g in range(NW)])[0, 1]) for k in ('c1', 'c2a', 'c2b', 'c3')}

body = f'''
<p class="lede">Scope reduced to one dispatch (rank {R}, GPU {4 + R}, uniform routing, 4,096 tokens, 16 SMs), one warp, one iteration of the token loop. Two questions: which phase of the iteration is the problematic one, and which warp exactly is the straggler. Both are answered from the per-token stamps already in the trace; nothing was re-run.</p>

<h2>1. The iteration and its phases</h2>
<p>Each dispatch warp runs the same loop for each of its tokens: wait for the previous token's stores to be accepted (C1, <code>tma_store_wait</code>), prepare the token (C2a: top-k loads and the destination-slot atomic issued; C2b: the atomic's return and the TMA load of the token landing in shared memory), then issue one bulk store per destination (C3). The trace stamps each boundary per token, and per lane the store issue and acceptance. A warp's finish is the acceptance of its last store.</p>

<h2>2. Which phase</h2>
{report.figure(fig1, f'Figure 1. Every iteration of three warps of the same dispatch, split into the four phases. Left: the straggler. Middle: the median warp. Right: the fastest warp.')}
<div class="finding"><b>C3, the store issue, in the first iterations.</b> The straggler spends {sp['c3'].sum():.0f} µs of its {fin[strag]:.0f} µs in C3 against {fp['c3'].sum():.0f} µs for the fastest warp; C1 is {sp['c1'].sum():.0f} vs {fp['c1'].sum():.0f} µs and C2 {sp['c2a'].sum() + sp['c2b'].sum():.0f} vs {fp['c2a'].sum() + fp['c2b'].sum():.0f} µs. The excess is front-loaded: the straggler's first six iterations spend {first6(sp):.0f} µs in C3 and the remaining twelve {rest(sp):.0f} µs; for the fastest warp the numbers are {first6(fp):.0f} and {rest(fp):.0f} µs. Across all 240 warps the median first-six C3 is {c3_all_first:.0f} µs and the median rest {c3_all_rest:.0f} µs. The correlation of a warp's finish with its total time per phase is {corr['c3']:.2f} for C3, {corr['c1']:.2f} for C1, {corr['c2a']:.2f} for C2a and {corr['c2b']:.2f} for C2b.</div>
{report.figure(fig2, 'Figure 2. C3 issue time of every iteration of every warp, warps sorted by finish time. The dark band in the first iterations is the start-of-kernel burst, when all 240 warps issue at once; it is darkest and longest for the warps that finish last, which sit on the slow TPCs.')}
<p>So within one iteration the problematic phase is the issue of the bulk stores, not the wait, not the load and not the atomic. The issue instruction blocks when the per-lane queue of about seven stores is full (E1); at the start of the kernel every warp fills that queue at once, and the queue of a slow TPC drains slowest, so its warps pay tens of µs per iteration until the burst clears, and never catch up.</p>

<h2>3. Which warp</h2>
<div class="finding"><b>The exact straggler is block {strag // 15}, dispatch warp {strag % 15}, on physical SM {smid[strag]}, finishing at {fin[strag]:.0f} µs with {sp['n']} tokens and {sp['stores']} stores.</b> It is not an anomaly but the predictable product of three structural facts. (1) Its block runs on a slow TPC (physical SM {smid[strag]}, paired with SM {smid[strag] ^ 1}, whose block finishes at the same level), and its fifteen warps all finish within about 10 µs of each other except three. (2) Those three, dispatch warps 3, 7 and 11, sit on warp scheduler 3 together with a notify warp that finishes early, so they share their scheduler with two other dispatch warps instead of three and finish {-dev3:.0f} µs before their block (Figure 3); the straggler is on scheduler {sched[strag]}. (3) It is dispatch warp 0, and the static token assignment gives warp 0 of every block an 18th token, 14 KB more work than its neighbours.</div>
{report.figure(fig3, 'Figure 3. Left: finish of every warp by block, coloured by warp scheduler. Right: each warp’s finish minus its block’s median, by scheduler: the three warps on scheduler 3 are consistently earlier.')}

<h2>4. What this says about the method</h2>
<ol>
<li><b>Granularity is sufficient.</b> One iteration of one warp is decomposed into four phases with 32 ns stamps at about 1 % cost; the straggler is identified by name (block, warp, physical SM, scheduler) and its excess is placed in one phase and one range of iterations.</li>
<li><b>The straggler is structural, at three levels.</b> TPC (placement, E7), warp scheduler share within the SM (the notify warps' placement), and the token assignment's remainder. Each is a property of the launch, not of the fabric, and each is predictable before the kernel runs once placement is known.</li>
<li><b>The remedy is local.</b> Stagger the first issues to avoid the start burst, assign the remainder tokens to scheduler-3 warps, give slow TPCs fewer tokens: all of them software, all testable with the same trace.</li>
<li><b>What is still not visible</b> is where the slow TPC's bytes wait after they leave the SM. The phase is known (C3), the location is known to the TPC, the cause beyond that is the dark segment.</li>
</ol>
'''
html = report.page('One iteration, one warp: where is the straggler?', 'Scope reduced to a single dispatch warp\'s token loop: which phase of the iteration is slow, and which warp exactly is last, from the existing trace.',
                   f'E7 follow-up · rank {R}, repeat {REP} of results/ratio1_smid.pt · {time.strftime("%Y-%m-%d")}', body)
open(os.path.join(HERE, 'one-iteration.html'), 'w').write(html); print('wrote one-iteration.html')
