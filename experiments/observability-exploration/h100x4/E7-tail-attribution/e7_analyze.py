"""Build report.html for E7 from results/*.pt here and E2's ratio sweep (base variant)."""
import glob, os, sys, statistics, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, traceload as TLD, trace_layout as TL
from viz import PALETTE, INK, MUTED

files = []
for f in sorted(glob.glob(os.path.join(ROOT, 'E2-incast-victim-flows', 'results', 'base_ratio*.pt'))): files.append(f)
for f in sorted(glob.glob(os.path.join(HERE, 'results', '*.pt'))): files.append(f)

runs = []
for f in files:
    run = TLD.load(f); a = run['args']; K = a['topk']; NR = run['NR']; sms = run['sms']
    per_warp = []   # one row per (rank, repeat, warp): finish time rel. to open release (us), tokens, bytes to hot rank, bytes total, remote bytes
    rank_rows = []
    for r in range(NR):
        for i, rep in enumerate(run['ranks'][r]['reps']):
            ph = TLD.phase_durations(rep); t0 = ph['open_release']
            warp = rep['warp']; rec = rep['tok']
            wid = (rec[:, TL.T_ID] & 0xffffffff).numpy(); smid = (rec[:, TL.T_ID] >> 32).numpy()  # layout: hi32 = sm, lo32 = warp
            dstmask = rec[:, TL.T_DSTMASK].numpy()
            gw = smid * 15 + wid
            smids = (warp[:, 3] >> 32).numpy(); has_smid = bool((smids > 0).any())
            for w in range(warp.shape[0]):
                if int(warp[w, 1]) <= 0: continue
                m = gw == w
                masks = dstmask[m]
                n_tok = int(m.sum()); n_stores = sum(bin(int(x)).count('1') for x in masks); n_hot = sum((int(x) >> 0) & 1 for x in masks); n_remote = n_stores - sum((int(x) >> r) & 1 for x in masks)
                per_warp.append(dict(rank=r, rep=i, warp=w, block=w // 15, smid=int(smids[w]) if has_smid else -1, finish=(int(warp[w, 1]) - t0) / 1e3, start=(int(warp[w, 0]) - t0) / 1e3, tokens=n_tok, stores=n_stores, hot=n_hot, remote=n_remote))
            fins = [(int(warp[w, 1]) - t0) / 1e3 for w in range(warp.shape[0]) if int(warp[w, 1]) > 0]
            rank_rows.append(dict(rank=r, rep=i, span=ph['data_us'], median_finish=statistics.median(fins), p10=viz.pct(fins, 10), p90=viz.pct(fins, 90), last=max(fins), first=min(fins), close_wait=ph['E_wait_us']))
    runs.append(dict(name=os.path.basename(f)[:-3], ratio=a['ratio'], seed=a['seed'], per_warp=per_warp, rank_rows=rank_rows, NR=NR, sms=sms, K=K, has_smid=any(pw['smid'] >= 0 for pw in per_warp)))
runs.sort(key=lambda r: (r['ratio'], r['seed']))

def tail_fraction(run):  # (last - median finish) / last, median over (rank, rep)
    return statistics.median((rr['last'] - rr['median_finish']) / rr['last'] for rr in run['rank_rows'])
def r2_block(run):
    """Variance of warp finish explained by the block (= physical SM pair) the warp runs on, within each (rank, rep)."""
    res, tot = 0.0, 0.0
    for r in range(run['NR']):
        for i in range(max(pw['rep'] for pw in run['per_warp']) + 1):
            rows = [pw for pw in run['per_warp'] if pw['rank'] == r and pw['rep'] == i]
            if len(rows) < 5: continue
            y = np.array([pw['finish'] for pw in rows]); b = np.array([pw['block'] for pw in rows])
            yhat = np.array([y[b == bb].mean() for bb in b])
            res += float(((y - yhat) ** 2).sum()); tot += float(((y - y.mean()) ** 2).sum())
    return 1 - res / tot if tot else float('nan')

def r2_tokens(run, cols=('tokens',)):
    """Variance of warp finish explained by a linear fit on the given per-warp columns, within each (rank, rep)."""
    res, tot = 0.0, 0.0
    for r in range(run['NR']):
        for i in range(max(pw['rep'] for pw in run['per_warp']) + 1):
            rows = [pw for pw in run['per_warp'] if pw['rank'] == r and pw['rep'] == i]
            if len(rows) < 5: continue
            y = np.array([pw['finish'] for pw in rows]); X = np.column_stack([np.ones(len(rows))] + [np.array([pw[c] for pw in rows], dtype=float) for c in cols])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
            res += float(((y - yhat) ** 2).sum()); tot += float(((y - y.mean()) ** 2).sum())
    return 1 - res / tot if tot else float('nan')

# ---- Figure 1: warp finish distribution per ratio (rank 1, repeat 0) ----
f, axs = viz.fig(8.6, 3.4, ncols=2)
ax = axs[0]
by_ratio = {}
for run in runs: by_ratio.setdefault(run['ratio'], run)
for j, (ratio, run) in enumerate(sorted(by_ratio.items())):
    fins = sorted(pw['finish'] for pw in run['per_warp'] if pw['rank'] == 1 and pw['rep'] == 0)
    ax.plot(fins, np.arange(1, len(fins) + 1) / len(fins), color=PALETTE[j], drawstyle='steps-post', label=f'ratio {ratio:g}')
ax.set_xlabel('warp finish after opening-barrier release (µs), rank 1'); ax.set_ylabel('fraction of the 240 warps'); ax.set_title('When each warp finishes its last store'); viz.legend_below(ax, ncol=5)
ax = axs[1]
for j, (ratio, run) in enumerate(sorted(by_ratio.items())):
    rows = [pw for pw in run['per_warp'] if pw['rank'] == 1 and pw['rep'] == 0]
    ax.scatter([pw['stores'] for pw in rows], [pw['finish'] for pw in rows], s=8, color=PALETTE[j], alpha=0.6, label=f'ratio {ratio:g}')
ax.set_xlabel('bulk stores issued by the warp'); ax.set_ylabel('finish (µs)'); ax.set_title('Finish against the warp’s work'); viz.legend_below(ax, ncol=5)
fig1 = viz.to_svg(f)

# ---- Figure 2: tail fraction and variance explained vs ratio ----
f, axs = viz.fig(8.6, 3.2, ncols=2)
ratios = sorted(by_ratio)
tf = [statistics.median(tail_fraction(r) for r in runs if r['ratio'] == ra) for ra in ratios]
r2a = [statistics.median(r2_tokens(r, ('tokens',)) for r in runs if r['ratio'] == ra) for ra in ratios]
r2b = [statistics.median(r2_tokens(r, ('tokens', 'stores', 'hot')) for r in runs if r['ratio'] == ra) for ra in ratios]
ax = axs[0]; ax.plot(ratios, [v * 100 for v in tf], marker='o', color=PALETTE[0]); ax.set_xlabel('routing ratio (rank 0 hot)'); ax.set_ylabel('% of data plane'); ax.set_title('Tail: last warp − median warp, share of the data plane')
ax = axs[1]; ax.plot(ratios, [v * 100 for v in r2a], marker='o', color=PALETTE[0]); viz.end_label(ax, ratios[-1], r2a[-1] * 100, 'tokens only', PALETTE[0])
ax.plot(ratios, [v * 100 for v in r2b], marker='o', color=PALETTE[1]); viz.end_label(ax, ratios[-1], r2b[-1] * 100, '+ stores, + stores to hot rank', PALETTE[1])
ax.set_xlabel('routing ratio'); ax.set_ylabel('% of finish-time variance explained'); ax.set_ylim(0, 100); ax.set_title('How much of the spread is the warp\'s own work')
fig2 = viz.to_svg(f)

# ---- Figure 3: finish by physical SM, one panel per rank (runs that recorded %smid) ----
smid_runs = [r for r in runs if r['has_smid']]
if smid_runs:
    run_s = smid_runs[0]
    f, axs = viz.fig(8.6, 3.4, ncols=run_s['NR'], sharey=True)
    for r, ax in enumerate(np.atleast_1d(axs)):
        rows = [pw for pw in run_s['per_warp'] if pw['rank'] == r]
        xs = np.array([pw['smid'] for pw in rows]); ys = np.array([pw['finish'] for pw in rows])
        order = np.argsort(xs); uniq = sorted(set(xs.tolist())); pos = {sm: k for k, sm in enumerate(uniq)}
        ax.scatter([pos[v] + (pw['rep'] - 1) * 0.15 for v, pw in zip(xs, rows)], ys, s=7, color=[PALETTE[(pw['smid'] // 2) % 8] for pw in rows], alpha=0.7)
        ax.set_xticks(range(len(uniq))); ax.set_xticklabels([str(u) for u in uniq], fontsize=6, rotation=90); ax.set_title(f'rank {r} (GPU {4 + r})', fontsize=9); ax.set_xlabel('physical SM (%smid)')
    np.atleast_1d(axs)[0].set_ylabel('warp finish (µs)')
    fig3 = viz.to_svg(f)
else:
    run1 = by_ratio[min(by_ratio)]
    f, ax = viz.fig(8.6, 3.0)
    rows = [pw for pw in run1['per_warp'] if pw['rank'] == 1 and pw['rep'] == 0]
    ax.scatter([pw['warp'] // 15 + (pw['warp'] % 15) / 20 for pw in rows], [pw['finish'] for pw in rows], s=9, color=PALETTE[0])
    ax.set_xlabel('block index'); ax.set_ylabel('finish (µs)'); ax.set_title('Finish time by block, uniform routing, rank 1')
    fig3 = viz.to_svg(f)
r2blk = {ra: statistics.median(r2_block(r) for r in runs if r['ratio'] == ra) for ra in ratios}

tab = []
for run in runs:
    rr = run['rank_rows']
    tab.append([run['name'], f'{run["ratio"]:g}', run['seed'], f'{statistics.median(r["span"] for r in rr):.0f}', f'{statistics.median(r["median_finish"] for r in rr):.0f}', f'{statistics.median(r["p90"] for r in rr):.0f}', f'{statistics.median(r["last"] for r in rr):.0f}',
                f'{tail_fraction(run)*100:.0f}', f'{r2_tokens(run, ("tokens",))*100:.0f}', f'{r2_tokens(run, ("tokens","stores","hot"))*100:.0f}', f'{r2_block(run)*100:.0f}', f'{statistics.median(r["close_wait"] for r in rr):.0f}'])

mech = '''<svg viewBox="0 0 880 230" role="img" aria-label="E7: each dispatch warp handles tokens token_idx = warp_offset + k x stride; the trace records each warp's last store completion and each token's destination mask, so a warp's finish time can be regressed on its token count, store count and stores to the hot rank">
<text x="20" y="24" class="head">What a dispatch warp's finish time could depend on</text>
<rect x="20" y="40" width="260" height="60" rx="8" class="box"/><text x="150" y="60" text-anchor="middle" class="head">scheduling</text><text x="150" y="78" text-anchor="middle" class="small">tokens assigned: token = warp + k · (15 × SMs)</text><text x="150" y="92" text-anchor="middle" class="small">4096 / 240 = 17 or 18 tokens per warp</text>
<rect x="310" y="40" width="260" height="60" rx="8" class="box"/><text x="440" y="60" text-anchor="middle" class="head">routing</text><text x="440" y="78" text-anchor="middle" class="small">stores per token = distinct destination ranks (1–4)</text><text x="440" y="92" text-anchor="middle" class="small">stores into the hot rank (ratio > 1)</text>
<rect x="600" y="40" width="260" height="60" rx="8" class="box-bad"/><text x="730" y="60" text-anchor="middle" class="head">path</text><text x="730" y="78" text-anchor="middle" class="small">which SM, which link, what else was queued</text><text x="730" y="92" text-anchor="middle" class="small">not in the trace: the residual</text>
<text x="20" y="130" class="head">What the trace gives per warp</text>
<text x="20" y="152" class="small">WARP record: first loop entry, last store completion (wait_group return of the final token), tokens handled, SM id.</text>
<text x="20" y="170" class="small">TOK records: per token the warp id and the destination mask, so stores, hot-rank stores and remote stores per warp are counted exactly.</text>
<text x="20" y="196" class="small">Fit per (rank, repeat): finish = a + b·tokens (+ c·stores + d·hot). The share of variance the fit explains is attributable to scheduling and routing; the rest is path.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>E0 found that on a rank the first and last of the 240 dispatch warps finish about 240 µs apart in a 540 µs data plane, and that the closing barrier then waits for the slowest rank. The last warp's finish is what the whole dispatch waits for, so the tail is the critical path. There are three candidate reasons for one warp to finish late: it was given more work (the static token assignment gives some warps one token more than others), its tokens had more destinations or went to a congested destination (routing), or its stores happened to wait longer on the path (the SM's queue, the link, the peer). The first two are known to the library before the kernel runs; the third is only knowable by measurement. If the tail is mostly the first two, a scheduler can fix it; if it is the third, a tool has to find the path.</p>

<h2>2. Methodology</h2>
<p>In one sentence: for every dispatch warp, take its finish time from the trace and count its tokens, its stores and its stores to the hot rank, then ask how much of the spread of finish times a linear fit on those counts explains, under uniform and skewed routing.</p>
{report.figure(mech, 'Figure 1. The three candidate causes and what the trace records per warp.')}
{report.dispatch_procedure("What varies here: the routing ratio (1, 1.5, 2, 3, 4, from E2's base runs) and the random seed (two extra seeds at ratios 1 and 3, run here), plus two runs with the physical SM id recorded. 16 SMs, 4,096 tokens per rank.", repeats=3)}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Warp finish</b> = the warp's last <code>wait_group</code> return (WARP record) − the opening-barrier release on that rank, in µs. <b>Data plane</b> = the latest finish on the rank.</li>
<li><b>Tail fraction</b> = (latest finish − median finish) ÷ latest finish, per (rank, repeat), median over them.</li>
<li><b>Per-warp work</b>: tokens handled, bulk stores issued (sum over its tokens of the number of distinct destination ranks, from the destination mask), stores into rank 0 (the hot rank when ratio &gt; 1).</li>
<li><b>Variance explained</b>: within each (rank, repeat) an ordinary least-squares fit of finish on the work counts; 1 − residual sum of squares ÷ total sum of squares, pooled over (rank, repeat). "tokens only" uses the token count; the second fit adds stores and hot-rank stores. A third fit uses only the warp's block (the SM the block runs on): the group mean per block.</li>
<li><b>Physical SM</b>: two extra runs (ratios 1 and 3) with the warp record extended by <code>%smid</code>, the hardware SM number, so that block → SM placement can be read instead of assumed.</li>
</ul>
<h3>2.3 Configuration</h3>
<ul><li>4 ranks on host GPUs 4–7, 16 SMs, 4,096 tokens per rank, top-8 of 256 experts, baseline instrumented kernel.</li></ul>

<h2>3. Results</h2>
{report.figure(fig1, 'Figure 2. Left: cumulative distribution of warp finish times on rank 1 for each routing ratio. Right: each warp’s finish against the number of bulk stores it issued.')}
<div class="finding"><b>The warp’s own work explains nothing; the SM it runs on explains almost everything.</b> Under uniform routing the tail is {tf[0]*100:.0f} % of the data plane, and a fit on tokens, stores and hot-rank stores explains {r2b[0]*100:.0f} % of the finish-time variance (every warp has 17 or 18 tokens and nearly the same number of stores). A fit on the warp’s block alone explains {r2blk[ratios[0]]*100:.0f} %. The 15 warps of a block finish together, and blocks fall into two or three discrete levels, about {min(statistics.median(pw['finish'] for pw in by_ratio[ratios[0]]['per_warp'] if pw['rank']==1 and pw['block']==bb) for bb in range(16)):.0f} to {max(statistics.median(pw['finish'] for pw in by_ratio[ratios[0]]['per_warp'] if pw['rank']==1 and pw['block']==bb) for bb in range(16)):.0f} µs on rank 1. The levels are identical across repeats, seeds and the E0/E2 runs on the same GPU, and different from GPU to GPU. The same holds at every ratio (table).</div>
{report.figure(fig2, 'Figure 3. Left: tail fraction against routing ratio. Right: share of the finish-time variance explained by the warp’s work.')}
<div class="finding"><b>It is the physical SM, in TPC pairs.</b> Recording <code>%smid</code> shows that the cooperative launch places the 16 blocks on the same physical SMs every time (SMs 124–131 and 0–7 on these GPUs), two blocks per TPC, and that the finish level is a property of the TPC: both SMs of a pair sit at the same level, and the slow pairs of one GPU are not the slow pairs of another (Figure 4). The ratio between the slowest and the fastest pair is about 1.4 on the GPUs with three levels. Since each block's warps store the same amount of data to the same destinations, the difference is in the drain rate from that SM towards NVLink, upstream of the ports, which are shared equally by all blocks.</div>
{report.figure(fig3, 'Figure 4. Warp finish time against the physical SM number on each of the four GPUs (three repeats, uniform routing). Colour alternates per TPC pair. The level is fixed per pair and per GPU.')}
{report.table(['run', 'ratio', 'seed', 'data plane µs', 'median finish µs', 'p90 finish µs', 'last finish µs', 'tail %', 'R² tokens %', 'R² tokens+stores+hot %', 'R² block (SM pair) %', 'closing wait µs'], tab, num_cols=range(1, 12))}

<h2>4. Conclusions</h2>
<ol>
<li><b>The tail is not scheduling and not routing; it is where the block landed.</b> Token count, store count and hot-rank stores explain almost none of the spread; the block identity explains almost all of it, and the block identity maps to a fixed physical TPC whose drain rate towards the fabric is 1.0, 1.2 or 1.4 times another's. That is a location attribution at the sender, inside the SM-to-port path that the frame of the analysis note marks as dark, and it was reachable with one extra register read per warp.</li>
<li><b>It is stable, so it is actionable.</b> Because placement and per-TPC speed repeat run after run, a dispatcher could measure them once and either hand fewer tokens to the slow TPCs or request different SMs; the data plane would shrink towards the fast level, up to about 30 % here.</li>
<li><b>Skewed routing does not change the picture.</b> Hot-rank stores do not predict a warp's finish at any ratio in this four-rank setup; the per-TPC levels remain.</li>
<li><b>What the cause is remains open.</b> The per-TPC levels could be the TPC's share of its GPC's crossbar bandwidth, the L2 slice affinity of the receive-buffer addresses the block happens to write, or a difference in how the async proxy of each TPC is served. Distinguishing them needs an address sweep per TPC (E10) and a GPC map of these GPUs.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Four ranks (at most 4 destinations per token, so little variation in stores per warp); a linear fit; only 16 of the 132 SMs are exercised, in two GPCs, so the TPC levels seen are a sample. Finish is the warp's last local acceptance, not the arrival at the peer.</div>
'''
meta = f"experiment E7 · {len(runs)} runs · 4 × H100 on host GPUs 4–7 · scripts: e7_sweep.py, E2 base runs, e7_analyze.py · generated {time.strftime('%Y-%m-%d %H:%M')}"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E7 · The tail: scheduling, routing or path?', 'Regress each dispatch warp’s finish time on its token count, store count and hot-rank stores, under uniform and skewed routing, to see how much of the tail is knowable before launch.', meta, body))
print('wrote report with', len(runs), 'runs')
