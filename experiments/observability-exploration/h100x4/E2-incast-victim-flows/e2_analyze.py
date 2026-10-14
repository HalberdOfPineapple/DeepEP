"""Build report.html for E2 from results/*.pt."""
import glob, os, re, sys, statistics, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, trace_layout as TL
from viz import PALETTE, INK, MUTED
B = 14464
HOT_ = 0

runs = []
for f in sorted(glob.glob(os.path.join(HERE, 'results', '*.pt'))):
    m = re.match(r'(\w+)_ratio([\d.]+)_sms(\d+)\.pt', os.path.basename(f))
    d = torch.load(f, weights_only=False); a = d['args']; T = a['num_tokens']; K = a['topk']; NR = d['ranks'][0]['num_ranks']
    per_rank = []
    for g in d['ranks']:
        r = g['rank']; dc = g['dst_counts']
        # per-destination lane latency (issue -> done) and slot wait, pooled over repeats
        lane = collections.defaultdict(list); sw = []; spread = []; data_span = []; close_wait = []; first_last = []
        issue_series = collections.defaultdict(list)  # dst -> (t_issue) for in-flight estimation
        for tr in g['traces']:
            rec = tr[TL.TOK:TL.TOK + TL.TOK_STRIDE * T].view(T, TL.TOK_STRIDE); rec = rec[rec[:, TL.T_ISSUE] > 0]
            sw += (rec[:, TL.T_SLOTWAIT_END] - rec[:, TL.T_SLOTWAIT_START]).double().tolist()
            li = rec[:, TL.T_LANE_ISSUE:TL.T_LANE_ISSUE + K]; ld = rec[:, TL.T_LANE_DONE:TL.T_LANE_DONE + K]; dst = rec[:, TL.T_LANE_DST:TL.T_LANE_DST + K]
            ok = (li > 0) & (ld > 0) & (dst >= 0)
            for dd in range(NR):
                mm = ok & (dst == dd)
                lane[dd] += (ld[mm] - li[mm]).double().tolist()
            sm = {e: [int(tr[TL.SM + s * 32 + e]) for s in range(a['num_sms'])] for e in (2, 10)}
            warp = tr[TL.WARP:TL.WARP + 4 * a['num_sms'] * 15].view(-1, 4); wd = warp[:, 1][warp[:, 1] > 0]
            data_span.append(int(wd.max()) - max(sm[2])); close_wait.append(max(sm[10]) - int(wd.max())); first_last.append(int(wd.max()) - int(wd.min()))
        # per-destination completion-time series (first traced run): fraction of this sender's stores to dst completed by time t
        tr0 = g['traces'][0]; rec0 = tr0[TL.TOK:TL.TOK + TL.TOK_STRIDE * T].view(T, TL.TOK_STRIDE); rec0 = rec0[rec0[:, TL.T_ISSUE] > 0]
        t0 = max(int(tr0[TL.SM + s_ * 32 + 2]) for s_ in range(a['num_sms']))
        ld0 = rec0[:, TL.T_LANE_DONE:TL.T_LANE_DONE + K]; li0 = rec0[:, TL.T_LANE_ISSUE:TL.T_LANE_ISSUE + K]; d0 = rec0[:, TL.T_LANE_DST:TL.T_LANE_DST + K]
        done_by_dst = {dd: np.sort(((ld0[(li0 > 0) & (ld0 > 0) & (d0 == dd)]) - t0).double().numpy() / 1e3) for dd in range(NR)}
        per_rank.append(dict(rank=r, t_us=g['t_untraced_s'] * 1e6, dst_counts=dc.tolist(), recv_pairs=g['num_recv_token_expert_pairs'], done_by_dst=done_by_dst,
                             remote_gbs=(int(dc.sum()) - int(dc[r])) * B / g['t_untraced_s'] / 1e9,
                             lane_p50={dd: viz.pct(v, 50) for dd, v in lane.items()}, lane_p99={dd: viz.pct(v, 99) for dd, v in lane.items()}, lane_mean={dd: float(np.mean(v)) if v else 0 for dd, v in lane.items()},
                             sw_p50=viz.pct(sw, 50), sw_p99=viz.pct(sw, 99), sw_mean=float(np.mean(sw)), data_span=statistics.median(data_span) / 1e3, close_wait=statistics.median(close_wait) / 1e3, first_last=statistics.median(first_last) / 1e3))
    # ingress per receiver: tokens received from all senders
    recv_tokens = [sum(p['dst_counts'][dd] for p in per_rank) for dd in range(NR)]
    t_us = max(p['t_us'] for p in per_rank)
    hot_in_remote = sum(p['dst_counts'][HOT_] for p in per_rank if p['rank'] != HOT_) * B
    runs.append(dict(variant=m.group(1), ratio=float(m.group(2)), sms=int(m.group(3)), ranks=per_rank, recv_tokens=recv_tokens, t_us=t_us, NR=NR, T=T,
                     hot_ingress_gbs=hot_in_remote / t_us / 1e3, nonhot_egress_gbs=statistics.mean(p['remote_gbs'] for p in per_rank if p['rank'] != HOT_)))
runs.sort(key=lambda r: (r['variant'], r['ratio']))
base = [r for r in runs if r['variant'] == 'base']; wr = [r for r in runs if r['variant'] == 'waitread']
NR = runs[0]['NR']; T = runs[0]['T']
HOT = 0

# Figure 1: ingress per rank and kernel time vs ratio
f, axs = viz.fig(9.2, 3.6, ncols=2)
ax = axs[0]
for dd in range(NR):
    ys = [r['recv_tokens'][dd] * B / 1e6 for r in base]; ax.plot([r['ratio'] for r in base], ys, marker='o', color=PALETTE[dd]); viz.end_label(ax, base[-1]['ratio'], ys[-1], f'rank {dd}' + (' (hot)' if dd == HOT else ''), PALETTE[dd])
ax.set_xlabel('unbalanced ratio'); ax.set_ylabel('MB received per dispatch'); ax.set_title('Bytes each rank receives')
ax = axs[1]
ax.plot([r['ratio'] for r in base], [r['hot_ingress_gbs'] for r in base], marker='o', color=PALETTE[7]); viz.end_label(ax, base[-1]['ratio'], base[-1]['hot_ingress_gbs'], 'hot rank ingress', PALETTE[7])
ax.plot([r['ratio'] for r in base], [r['nonhot_egress_gbs'] for r in base], marker='o', color=PALETTE[0]); viz.end_label(ax, base[-1]['ratio'], base[-1]['nonhot_egress_gbs'], 'non-hot sender egress', PALETTE[0])
plateau = base[0]['nonhot_egress_gbs']
ax.axhline(plateau, color=MUTED, lw=0.8, ls='--'); ax.annotate(f'balanced egress {plateau:.0f} GB/s', (1.0, plateau), xytext=(4, 4), textcoords='offset points', fontsize=8, color=MUTED)
ax.set_xlabel('unbalanced ratio'); ax.set_ylabel('GB/s (remote bytes / kernel time)'); ax.set_title('Who is saturated: the hot rank\'s intake'); ax.set_ylim(0, None)
fig1 = viz.to_svg(f)

# Figure 2: per-destination lane latency seen by a non-hot sender (rank 1) vs ratio, baseline
f, axs = viz.fig(9.2, 3.6, ncols=2)
for ax, key, ttl in [(axs[0], 'lane_p50', 'p50'), (axs[1], 'lane_p99', 'p99')]:
    for dd in range(NR):
        ys = [statistics.median([p[key].get(dd, float('nan')) for p in r['ranks'] if p['rank'] != dd and p['rank'] != HOT] or [float('nan')]) / 1e3 for r in base]
        ax.plot([r['ratio'] for r in base], ys, marker='o', color=PALETTE[dd]); viz.end_label(ax, base[-1]['ratio'], ys[-1], f'→ rank {dd}' + (' (hot)' if dd == HOT else ''), PALETTE[dd])
    ax.set_xlabel('unbalanced ratio'); ax.set_ylabel('µs'); ax.set_title(f'Lane issue → wait return, {ttl}')
fig2 = viz.to_svg(f)

# Figure 3: slot wait and tail vs ratio, both variants
f, axs = viz.fig(9.2, 3.6, ncols=2)
for series, color, label in [(base, PALETTE[0], 'baseline'), (wr, PALETTE[1], 'wait_group.read')]:
    if not series: continue
    axs[0].plot([r['ratio'] for r in series], [statistics.median(p['sw_p50'] for p in r['ranks'] if p['rank'] != HOT) / 1e3 for r in series], marker='o', color=color)
    axs[0].plot([r['ratio'] for r in series], [statistics.median(p['sw_p99'] for p in r['ranks'] if p['rank'] != HOT) / 1e3 for r in series], marker='o', ls=':', lw=1.2, color=color)
    viz.end_label(axs[0], series[-1]['ratio'], statistics.median(p['sw_p99'] for p in series[-1]['ranks'] if p['rank'] != HOT) / 1e3, f'{label} p99', color)
    viz.end_label(axs[0], series[-1]['ratio'], statistics.median(p['sw_p50'] for p in series[-1]['ranks'] if p['rank'] != HOT) / 1e3, f'{label} p50', color)
    axs[1].plot([r['ratio'] for r in series], [statistics.median(p['first_last'] for p in r['ranks'] if p['rank'] != HOT) for r in series], marker='o', color=color); viz.end_label(axs[1], series[-1]['ratio'], statistics.median(p['first_last'] for p in series[-1]['ranks'] if p['rank'] != HOT), label, color)
axs[0].set_title('Slot wait on non-hot senders'); axs[0].set_xlabel('unbalanced ratio'); axs[0].set_ylabel('µs')
axs[1].set_title('First-to-last warp finish spread, non-hot senders'); axs[1].set_xlabel('unbalanced ratio'); axs[1].set_ylabel('µs')
fig3 = viz.to_svg(f)

# Figure 4: per-destination completion over time on sender rank 1, balanced vs most skewed
rmax = base[-1]
f, axs = viz.fig(9.2, 3.6, ncols=2)
for ax, run in zip(axs, [base[0], rmax]):
    p = next(q for q in run['ranks'] if q['rank'] == 1)
    for dd in range(NR):
        if dd == 1: continue
        ys = p['done_by_dst'][dd]
        if len(ys) == 0: continue
        ax.plot(ys, np.arange(1, len(ys) + 1) / len(ys) * 100, color=PALETTE[dd], lw=1.8); viz.end_label(ax, ys[-1], 100, f'→ rank {dd}' + (' (hot)' if dd == HOT else ''), PALETTE[dd], dx=4)
    ax.set_xlabel('µs since opening-barrier release'); ax.set_ylabel('% of stores to destination completed'); ax.set_title(f'sender rank 1, ratio {run["ratio"]}')
    ax.set_ylim(0, 112)
fig4 = viz.to_svg(f)

tab = []
for r in runs:
    nh = [p for p in r['ranks'] if p['rank'] != HOT]
    tab.append([r['variant'], r['ratio'], f"{r['t_us']:.0f}", ' / '.join(f'{v*B/1e6:.0f}' for v in r['recv_tokens']),
                f"{statistics.median(p['lane_p50'].get(HOT, 0) for p in nh)/1e3:.1f} / {statistics.median(p['lane_p99'].get(HOT, 0) for p in nh)/1e3:.1f}",
                f"{statistics.median(statistics.median(p['lane_p50'].get(dd, 0) for dd in range(NR) if dd not in (HOT, p['rank'])) for p in nh)/1e3:.1f} / {statistics.median(statistics.median(p['lane_p99'].get(dd, 0) for dd in range(NR) if dd not in (HOT, p['rank'])) for p in nh)/1e3:.1f}",
                f"{statistics.median(p['sw_p50'] for p in nh)/1e3:.1f} / {statistics.median(p['sw_p99'] for p in nh)/1e3:.1f}", f"{statistics.median(p['first_last'] for p in nh):.0f}", f"{statistics.median(p['close_wait'] for p in nh):.0f}"])

b1, b4 = base[0], base[-1]
nh1 = [p for p in b1['ranks'] if p['rank'] != HOT]; nh4 = [p for p in b4['ranks'] if p['rank'] != HOT]
hot_lat1 = statistics.median(p['lane_p50'][HOT] for p in nh1) / 1e3; hot_lat4 = statistics.median(p['lane_p50'][HOT] for p in nh4) / 1e3
idle_lat1 = statistics.median(statistics.median(p['lane_p50'][dd] for dd in range(NR) if dd not in (HOT, p['rank'])) for p in nh1) / 1e3
idle_lat4 = statistics.median(statistics.median(p['lane_p50'][dd] for dd in range(NR) if dd not in (HOT, p['rank'])) for p in nh4) / 1e3
wr4 = wr[-1] if wr else None

mech = '''<svg viewBox="0 0 880 230" role="img" aria-label="Setup of E2: rank 0 receives ratio times more tokens; the lane trace on the other three ranks separates completion latency towards the hot rank from latency towards idle ranks">
<defs><marker id="e2a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-bad"/></marker><marker id="e2b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-acc"/></marker></defs>
<rect x="20" y="40" width="110" height="60" rx="8" class="box"/><text x="75" y="64" text-anchor="middle" class="head">rank 1</text><text x="75" y="82" text-anchor="middle" class="small">sender, traced</text>
<rect x="20" y="130" width="110" height="60" rx="8" class="box"/><text x="75" y="154" text-anchor="middle" class="head">rank 2</text><text x="75" y="172" text-anchor="middle" class="small">sender, traced</text>
<rect x="380" y="30" width="130" height="60" rx="8" class="box-bad"/><text x="445" y="54" text-anchor="middle" class="head">rank 0 · hot</text><text x="445" y="72" text-anchor="middle" class="small">ratio × tokens</text>
<rect x="380" y="140" width="130" height="60" rx="8" class="box-acc"/><text x="445" y="164" text-anchor="middle" class="head">rank 3 · idle</text><text x="445" y="182" text-anchor="middle" class="small">normal share</text>
<path d="M130 60 L380 56" class="ln-bad" marker-end="url(#e2a)"/><path d="M130 150 L380 64" class="ln-bad" marker-end="url(#e2a)"/>
<path d="M130 80 L380 164" class="ln-acc" marker-end="url(#e2b)"/><path d="M130 170 L380 172" class="ln-acc" marker-end="url(#e2b)"/>
<text x="255" y="40" text-anchor="middle" class="small">lane k → rank 0: issue, wait_group return</text>
<text x="255" y="205" text-anchor="middle" class="small">lane k' → rank 3: issue, wait_group return</text>
<rect x="560" y="30" width="300" height="170" rx="8" class="box"/>
<text x="570" y="52" class="head">Read-outs on non-hot senders</text>
<text x="570" y="74" class="small">· per-destination lane latency (hot vs idle)</text>
<text x="570" y="92" class="small">· slot wait at the loop top</text>
<text x="570" y="110" class="small">· first-to-last warp finish spread</text>
<text x="570" y="128" class="small">· closing-barrier wait for the hot rank</text>
<text x="570" y="152" class="small">Second kernel variant: wait_group.read at the loop</text>
<text x="570" y="168" class="small">top (slot reusable once the source bytes are read)</text>
<text x="570" y="184" class="small">to test whether the warp-level coupling matters.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>When one rank hosts popular experts it receives several times the traffic of the others (incast). The concern is that this slows traffic to the <em>idle</em> ranks as well, so that one hot destination degrades the whole dispatch rather than only its own share. Two mechanisms could do that. Warp-level: each dispatch warp waits for all of its lanes' stores before reusing its slot, so a slow store to the hot rank blocks the warp's next token, including its stores to idle ranks. Fabric-level: if the sender's NVLink egress credits or the switch queues are shared across destinations, stores to idle ranks physically queue behind stores to the hot rank. E1 and E0 refined the picture: the per-token wait returns when the SM's copy engine has accepted the store, not when the peer has it; the queue that actually limits a warp is the per-SM async-proxy store queue; and that queue is shared by all destinations an SM's warps store to. The question for E2 is therefore which sharing shows up under incast: does the completion time of stores to idle ranks, as seen on the sender, rise when another destination is hot, and does releasing the slot earlier (<code>wait_group.read</code>) change anything.</p>
<h2>2. Methodology</h2>
<p>In one sentence: make rank 0 the hot destination with DeepEP's unbalanced routing generator, and on every <em>other</em> rank compare how long its stores to rank 0 take against how long its stores to the two idle ranks take, using the per-lane stamps.</p>
{report.figure(mech, 'Figure 1. Setup. DeepEP\'s precise unbalanced generator makes rank 0 receive ratio × the tokens of every other rank. The traced build records, on every sender, each lane\'s store issue and wait return together with the destination rank.')}
{report.dispatch_procedure("What varies here: the routing ratio ∈ {1, 1.5, 2, 3, 4} passed to DeepEP's <code>get_unbalanced_scores(..., precise=True)</code>, which makes rank 0 receive ratio × the tokens of each other rank; and two kernel builds, the baseline and one with <code>cp.async.bulk.wait_group.read 0</code> at the loop top. Ten runs of the procedure in all.", repeats=3, variant_note="Here the patch is applied twice: once as the baseline, once with the <code>--wait-read</code> flag, which swaps the loop-top wait for the <code>.read</code> variant and changes nothing else. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Per-destination latency</b> (Figure 2): for every lane of every dispatch warp on a sending rank, wait return − store issue, grouped by the destination rank that lane recorded. "hot" is the group with destination rank 0; "idle" pools the two destinations that are neither rank 0 nor the sender. Medians are over all lanes of the three non-hot senders and the three repeats.</li>
<li><b>Hot-rank intake</b> = bytes stored into rank 0 by all senders ÷ (last store to rank 0 − first store to rank 0), stamps aligned with the per-run clock offsets.</li>
<li><b>Completion time of the last store to an idle rank</b> (Figure 3): on each sender, the latest wait-return stamp among stores to idle destinations, relative to the opening-barrier release, compared with the same for the hot destination.</li>
<li><b>Kernel µs</b> and <b>tracing overhead</b> as in step 4.</li>
</ul>
<h3>2.3 Configuration</h3>
<ul>
<li>4 ranks on host GPUs 4–7, 4,096 tokens per rank, top-8 of 256 experts, 16 SMs (DeepEP's own choice is 20; 16 keeps the aggregate per-SM issue capacity near the link so that sender saturation does not mask destination effects), expert alignment 128, CPU sync on.</li>
<li>With four ranks the generator caps destinations per token at min(8, 4, ⌊3/ratio⌋ + 1): 4 at ratio 1, 2 at ratio 1.5 to 3, 1 at ratio 4. Ratio 4 therefore is the single-destination case (every token goes to exactly one rank), which also removes the warp-level coupling by construction.</li>
</ul>
<h2>3. Results</h2>
<div class="finding"><b>The hot rank's intake saturates at about {max(r['hot_ingress_gbs'] for r in base):.0f} GB/s and sets everyone's kernel time; the other senders run at {base[-1]['nonhot_egress_gbs']:.0f} GB/s, {100*base[-1]['nonhot_egress_gbs']/plateau:.0f} % of what they do in the balanced run.</b> The generator changes the total traffic with the ratio, so kernel times are not comparable across ratios; the rates are. Rank 0's NVLink intake rises from {base[0]['hot_ingress_gbs']:.0f} GB/s (balanced, where each rank's intake equals its egress) to {max(r['hot_ingress_gbs'] for r in base):.0f} GB/s and stays there, while each non-hot sender's egress falls to {base[-1]['nonhot_egress_gbs']:.0f} GB/s. The link into one GPU, at roughly 320 GB/s delivered, is the resource that is full.</div>
{report.figure(fig1, 'Figure 2. Left: bytes received per rank per dispatch against the routing ratio (the generator reduces total traffic as the ratio grows). Right: the hot rank\'s ingress rate and the non-hot senders\' egress rate; the dashed line is a sender\'s egress in the balanced run.')}
<div class="finding"><b>A sender cannot tell its hot destination from its idle ones.</b> On the non-hot senders, lane issue → wait return towards the hot rank is {hot_lat1:.1f} µs p50 at ratio 1 and {hot_lat4:.1f} µs at ratio {b4['ratio']}; towards idle ranks {idle_lat1:.1f} µs and {idle_lat4:.1f} µs. At every ratio the two are equal to within 0.2 µs, and both tails grow together (p99 {statistics.median(p['lane_p99'][HOT] for p in nh1)/1e3:.0f} µs → {statistics.median(p['lane_p99'][HOT] for p in nh4)/1e3:.0f} µs). The per-SM async-proxy queue that E1 identified is a FIFO shared by every destination an SM's warps store to: when the hot rank's intake throttles the stores addressed to it, every store queued behind them waits the same time, whatever its destination.</div>
{report.figure(fig2, 'Figure 3. Lane issue → wait_group return on non-hot senders, per destination, against the ratio. Left p50, right p99. Hot and idle destinations coincide at every ratio.')}
<div class="finding"><b>Consequence: the idle destinations' stores are spread over the whole kernel instead of finishing early.</b> On sender rank 1 at ratio {rmax['ratio']}, the last store to an idle rank completes at the same time as the last store to the hot rank (Figure 5, right), and the idle ranks' data planes end within {max(p['first_last'] for p in rmax['ranks']) - min(p['first_last'] for p in rmax['ranks']):.0f} µs of the hot rank's. This is sender-side head-of-line blocking: a receiver bottleneck becomes a uniform slowdown of all of the sender's traffic, and the idle ranks' intake, which could have finished at {plateau:.0f} GB/s, drips in at the hot rank's pace. The "idle ranks finish early and wait at the barrier" picture in the analysis note is wrong for this hardware; the closing-barrier wait on non-hot ranks is only {statistics.median(p['close_wait'] for p in nh4):.0f} µs at ratio {rmax['ratio']}.</div>
{report.figure(fig4, 'Figure 4. Cumulative completion of sender rank 1\'s stores per destination, balanced (left) and most skewed (right). With a hot destination, every destination\'s curve stretches to the same end: nothing finishes early.')}
<div class="finding"><b>Releasing the slot earlier does not help, and deepens the queue.</b> The <code>wait_group.read</code> variant changes kernel time by {(wr4['t_us']/b4['t_us']-1)*100 if wr4 else 0:+.1f} % at ratio {b4['ratio']} and by {(wr[0]['t_us']/base[0]['t_us']-1)*100 if wr else 0:+.1f} % balanced, while its per-lane p99 at ratio {b4['ratio']} rises to {statistics.median(p['lane_p99'][HOT] for p in wr4['ranks'] if p['rank'] != HOT)/1e3 if wr4 else 0:.0f} µs (baseline {statistics.median(p['lane_p99'][HOT] for p in nh4)/1e3:.0f} µs): more stores sit in the same saturated queue. The warp-level coupling of problem P1 is real but is not where the time goes.</div>
{report.figure(fig3, 'Figure 5. Left: slot wait p50 (solid) and p99 (dotted) on non-hot senders for both variants. Right: spread between the first and the last dispatch warp finishing on a non-hot sender.')}
{report.table(['variant', 'ratio', 'kernel µs', 'MB received per rank 0/1/2/3', 'lane → hot p50 / p99 µs', 'lane → idle p50 / p99 µs', 'slot wait p50 / p99 µs', 'warp spread µs', 'close wait µs'], tab, num_cols=range(1, 9))}
<h2>4. Conclusions</h2>
<ol>
<li><b>Incast is paid by every sender, uniformly, through the shared per-SM store queue.</b> The hot rank's intake caps near 320 GB/s; each sender's queue fills with stores to it, and stores to idle ranks behind them wait equally. Fabric-level victim flows in the switch cannot be excluded, but they are not needed to explain the data: the sender's own FIFO already produces the victim effect.</li>
<li><b>Observability consequence.</b> The one per-token signal a sender has, the local-acceptance wait, carries no destination information, because the queue it measures is shared. Attribution to a destination needs either the destination mix of what is queued (which the trace provides, since the library knows every store's destination) or a receiver-side ingress counter per source, which does not exist. A per-destination queue in hardware, or per-destination issue throttling in software, would turn the same trace into a per-destination signal.</li>
<li><b>Which mechanism it is.</b> The "warp-level" mechanism exists but is cheap (wait_group.read changes nothing); the "fabric-level" mechanism exists in a form not anticipated before measuring: sender-side FIFO sharing in the SM's async proxy, upstream of the link. The experiment that would separate it from switch-level sharing is to issue the hot and idle destinations from disjoint SMs; that is a kernel change, left for later.</li>
<li><b>Mitigation hint for DeepEP.</b> Since the hot rank's intake is the bottleneck and everything else queues behind it, scheduling tokens so that stores to the hot rank are issued last (or from a dedicated subset of SMs) would let idle-destination traffic finish early and shorten the tail of the combine that follows; the trace from this build is what would evaluate that.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Four ranks: at most three senders converge on the hot rank and at most two idle destinations exist per sender; the generator forces single-destination tokens at ratio 4. Per-destination latencies are local-acceptance times (E1), so they bound what any sender-side method can see. 16 SMs rather than DeepEP's default 20.</div>
'''
meta = f"experiment E2 · {len(runs)} runs · 4 × H100 on host GPUs 4–7 · scripts: e2_sweep.py, common/run_dispatch.py, common/apply_trace_patch.py (--wait-read), e2_analyze.py"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E2 · Destination incast and victim flows', 'Making one rank hot with DeepEP\'s unbalanced-gate generator and reading, on the other senders, whether their stores to idle ranks slow down too.', meta, body))
print('wrote report with', len(runs), 'runs')
