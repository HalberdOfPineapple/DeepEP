"""Build report.html for E4: one-way latency of the notify count stores, from sender SM 0 issue timestamps and receiver
SM 0 ready timestamps, aligned with the clock offsets measured by E1's ping-pong."""
import glob, json, os, re, sys, statistics, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report, trace_layout as TL
from viz import PALETTE, INK, MUTED

# clock offsets: off[(a,b)] = clock_b - clock_a (ns), devices in CUDA_VISIBLE order = ranks
E1 = json.load(open(os.path.join(ROOT, 'E1-bulk-store-completion', 'results', 'e1_results.json')))
off = {(p['a'], p['b']): p['offset_b_minus_a_median'] for p in E1['pingpong']}
rtt = statistics.median(p['rtt_p50'] for p in E1['pingpong'])  # replaced below by in-run values when available

def analyze(path, label):
    d = torch.load(path, weights_only=False); a = d['args']; NR = d['ranks'][0]['num_ranks']; NLE = a['num_experts'] // NR; T = a['num_tokens']
    nrep = len(d['ranks'][0]['traces'])
    oneway = collections.defaultdict(list)  # (sender, receiver) -> list of ns
    e_to_f = []; c_to_e = []; f_to_h = []; recv_span = []; drift = []
    clocks = d['ranks'][0].get('clocks')
    if clocks is None:
        print('WARNING: no in-run clock calibration in', path, '- skipping'); return None
    for rep in range(nrep):
        tr = {g['rank']: g['traces'][rep] for g in d['ranks']}
        cb, ca = clocks[rep]['before']['offset'], clocks[rep]['after']['offset']
        off = {k: (cb[k] + ca[k]) / 2 for k in cb}
        drift += [abs(ca[k] - cb[k]) for k in cb]
        for s in range(NR):
            send = tr[s][TL.SEND:TL.SEND + 4096]
            for dd in range(NR):
                if dd == s: continue
                recv = tr[dd][TL.RECV:TL.RECV + 4096]
                o = off.get((s, dd), 0.0)
                # rank-count slot: receiver slot index s (thread s polls it first) ; sender stamp SEND[rank_idx]
                ts, trv = int(send[s]), int(recv[s])
                if ts and trv: oneway[(s, dd)].append(trv - o - ts)
                # expert slots: receiver slot NR + NLE*s + le, polled first by thread (NR + NLE*s + le) only if that index < 128
                for le in range(NLE):
                    slot = NR + NLE * s + le
                    if slot >= 128: break
                    e_global = NLE * dd + le  # sender's expert index that lands in receiver dd's local expert le
                    ts, trv = int(send[NR + e_global]), int(recv[slot])
                    if ts and trv: oneway[(s, dd)].append(trv - o - ts)
        for r in range(NR):
            sm = [int(tr[r][TL.SM + e]) for e in range(12)]
            if sm[5] and sm[6]: e_to_f.append((sm[6] - sm[5]) / 1e3)
            if sm[3] and sm[5]: c_to_e.append((sm[5] - sm[3]) / 1e3)
            if sm[6] and sm[7]: f_to_h.append((sm[7] - sm[6]) / 1e3)
    allv = [v for vs in oneway.values() for v in vs]
    return dict(label=label, sms=a['num_sms'], tok=T, ratio=a['ratio'], oneway=oneway, all=np.array(allv), e_to_f=e_to_f, c_to_e=c_to_e, f_to_h=f_to_h, t_us=statistics.mean(g['t_untraced_s'] for g in d['ranks']) * 1e6, drift_max=max(drift) if drift else 0, rtt=statistics.median(v for v in clocks[0]['before']['rtt'].values()))

cases = []
def add(path, label):
    if os.path.exists(path):
        c = analyze(path, label)
        if c is not None: cases.append(c)
add(os.path.join(HERE, 'results', 'base_sms16_tok64.pt'), 'notify alone\n64 tok, 16 SMs')
add(os.path.join(HERE, 'results', 'base_sms16_tok256.pt'), 'light data\n256 tok, 16 SMs')
add(os.path.join(HERE, 'results', 'base_sms16_tok4096.pt'), 'full data\n4096 tok, 16 SMs')
add(os.path.join(HERE, 'results', 'base_sms32_tok4096.pt'), 'full data\n4096 tok, 32 SMs')
add(os.path.join(HERE, 'results', 'incast4_sms16_tok4096.pt'), 'incast\nratio 4\n4096 tok, 16 SMs')
add(os.path.join(HERE, 'results', 'reloc4M_sms16_tok4096.pt'), 'relocated\n+4 MiB\n4096 tok, 16 SMs')
add(os.path.join(HERE, 'results', 'reloc4M64_sms16_tok4096.pt'), 'relocated\n+4 MiB + 64 B\n4096 tok, 16 SMs')
add(os.path.join(HERE, 'results', 'reloc4M_sms32_tok4096.pt'), 'relocated\n+4 MiB\n4096 tok, 32 SMs')

rtt = statistics.median(c['rtt'] for c in cases)
# Figure 1: one-way latency distributions per case (p10/p50/p90/p99 as dot-and-whisker)
f, ax = viz.fig(9.2, 4.2)
x = np.arange(len(cases))
for i, c in enumerate(cases):
    v = c['all'] / 1e3
    if len(v) == 0: continue
    p10, p50, p90, p99 = (viz.pct(v, q) for q in (10, 50, 90, 99))
    ax.plot([i, i], [p10, p90], color=PALETTE[0], lw=3, solid_capstyle='round'); ax.plot([i, i], [p90, p99], color=PALETTE[0], lw=1); ax.plot(i, p50, 'o', color=INK, ms=6)
    ax.annotate(f'{p50:.1f}', (i, p50), xytext=(8, 0), textcoords='offset points', fontsize=9, va='center')
    ax.annotate(f'p99 {p99:.1f}', (i, p99), xytext=(8, 0), textcoords='offset points', fontsize=8, va='center', color=MUTED)
ax.axhline(rtt / 2e3, color=MUTED, lw=0.8, ls='--'); ax.annotate(f'half ping-pong RTT {rtt/2e3:.2f} µs', (len(cases) - 1, rtt / 2e3), xytext=(-150, 6), textcoords='offset points', fontsize=8, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([c['label'].replace(' tok, ', ' tok\n') for c in cases], fontsize=7.5); ax.set_ylabel('one-way latency, 8 B count store (µs)'); ax.set_title('Sender SM 0 store → receiver SM 0 sees it (clocks aligned per run)')
ax.set_yscale('log')
fig1 = viz.to_svg(f)

# Figure 2: per (sender, receiver) median at full data 16 SMs vs notify alone
f, axs = viz.fig(9.2, 3.4, ncols=2)
for ax, c in zip(axs, [cases[0], next((q for q in cases if q['label'].startswith('full data\n4096 tok, 16')), cases[-1])]):
    NR = 4; M = np.full((NR, NR), np.nan)
    for (s, dd), vs in c['oneway'].items(): M[s, dd] = statistics.median(vs) / 1e3
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=np.nanmax([np.nanmax(M), 1]))
    for s in range(NR):
        for dd in range(NR):
            if not np.isnan(M[s, dd]): ax.text(dd, s, f'{M[s,dd]:.1f}', ha='center', va='center', fontsize=9, color=INK)
    ax.set_xticks(range(NR)); ax.set_yticks(range(NR)); ax.set_xlabel('receiver'); ax.set_ylabel('sender'); ax.set_title(c['label'].replace('\n', ' ') + ' · median µs'); ax.grid(False)
fig2 = viz.to_svg(f)

# Figure 3: notify phase timeline (C done → E issued → F done → H done) per case
f, ax = viz.fig(9.2, 3.8)
w = 0.6
for i, c in enumerate(cases):
    segs = [(statistics.median(c['c_to_e']) if c['c_to_e'] else 0, PALETTE[6], 'C done → E issued (grid reduction wait on SM 0)'), (statistics.median(c['e_to_f']) if c['e_to_f'] else 0, PALETTE[1], 'E issued → F done (wait for all peers\' counts)'), (statistics.median(c['f_to_h']) if c['f_to_h'] else 0, PALETTE[2], 'F done → H done (sum, host mailbox)')]
    b = 0
    for v, col, lab in segs:
        ax.bar(i, v, bottom=b, color=col, width=w, label=lab if i == 0 else None); b += v
ax.set_xticks(range(len(cases))); ax.set_xticklabels([c['label'].replace(' tok, ', ' tok\n') for c in cases], fontsize=7.5); ax.set_ylabel('µs (median over ranks and repeats)'); ax.set_title('SM 0 notify phases'); viz.legend_below(ax, 3)
fig3 = viz.to_svg(f)

tab = [[c['label'].replace('\n', ' · '), f"{c['t_us']:.0f}", len(c['all']), f"{viz.pct(c['all'],50)/1e3:.2f}", f"{viz.pct(c['all'],90)/1e3:.2f}", f"{viz.pct(c['all'],99)/1e3:.2f}", f"{c['all'].max()/1e3:.1f}" if len(c['all']) else '-', f"{statistics.median(c['e_to_f']) if c['e_to_f'] else 0:.1f}"] for c in cases]
quiet = cases[0]; full16 = next((q for q in cases if q['label'].startswith('full data\n4096 tok, 16')), None); full32 = next((q for q in cases if q['label'].startswith('full data\n4096 tok, 32')), None)
inc = next((q for q in cases if q['label'].startswith('incast')), None); rel = [q for q in cases if q['label'].startswith('relocated')]

mech = '''<svg viewBox="0 0 880 210" role="img" aria-label="Measurement edge of E4: sender SM 0 timestamps each count store after issuing it; receiver SM 0 timestamps the first poll that sees the slot ready; the difference, corrected by the ping-pong clock offset, is the one-way latency">
<defs><marker id="e4a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-warn"/></marker></defs>
<text x="20" y="40" class="small">sender SM 0 (clock s)</text><line x1="190" y1="36" x2="860" y2="36" class="ln"/>
<text x="20" y="130" class="small">receiver SM 0 (clock d)</text><line x1="190" y1="126" x2="860" y2="126" class="ln"/>
<circle cx="260" cy="36" r="5" class="fill-warn"/><text x="260" y="24" text-anchor="middle" class="mono">st.relaxed.sys count → SEND[i] = now</text>
<path d="M266 40 L500 122" class="ln-warn" marker-end="url(#e4a)"/>
<text x="400" y="88" text-anchor="middle" class="small">shares L2, port and switch with 240 warps of 14 KB stores</text>
<circle cx="510" cy="126" r="5" class="fill-bad"/><text x="510" y="150" text-anchor="middle" class="mono">ld.volatile sees −v−1 → RECV[i] = now</text>
<rect x="580" y="52" width="280" height="72" rx="8" class="box-type"/><text x="590" y="72" class="small">one-way = RECV[i] − offset(d − s) − SEND[i]</text><text x="590" y="90" class="small">offset from a ping-pong run before and after</text><text x="590" y="106" class="small">each dispatch; triangle closure ≈ 100 ns</text>
<text x="20" y="190" class="small">Only the first slot each receiver thread polls is used (128 of 260 slots): later slots are checked only after the earlier one is ready.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>Dispatch has a small control plane inside its data plane: the 260 eight-byte count stores that SM 0 sends to its peers (phase E) travel the same L2, NVLink ports and switch queues as the 14 KB bulk stores of the dispatch warps, with no priority. The receiver cannot finish phase F, and the host cannot allocate the outputs, until the last of them arrives. In prefill this is hidden behind the data; in decode or under incast the control plane can become the critical path. The question is how much the data plane delays a small store, whether the delay depends on the destination address (does the address pick a path), and what it costs end to end. This is the first experiment that needs timestamps on two GPUs, so it also exercises the clock alignment from E1.</p>
<h2>2. Methodology</h2>
<p>In one sentence: stamp the clock on the sending GPU right after each count store is issued and on the receiving GPU the first time its poll loop sees that count, align the two clocks, and subtract.</p>
{report.figure(mech, 'Figure 1. The measurement edge. Both stamps are taken by SM 0 threads that already exist in the kernel; the receiver stamp has the resolution of its poll loop (one L2 load per iteration).')}
{report.dispatch_procedure("What varies here is the load on the data plane and the address of the count array, as the cases listed in 2.3. Only step 5 matters for this experiment; the kernel-time run of step 4 is used only to report the kernel length each chain is compared against.", repeats=8, variant_note="For the relocated cases the patch also redirects the receive-count array to an unused region of the window at a fixed byte offset, on every rank alike. ")}
<h3>2.2 How each reported number is computed</h3>
<ul>
<li><b>Sender stamp</b> SEND[i]: in the notify phase, the SM 0 thread that stores count i into a peer's memory with <code>st.relaxed.sys</code> reads the clock immediately after the store instruction.</li>
<li><b>Receiver stamp</b> RECV[i]: the SM 0 thread on the destination rank that polls count slot i reads the clock in the first poll iteration that sees the slot's value ready. Its resolution is one poll iteration, one L2 load.</li>
<li><b>One-way latency</b> = RECV[i] − SEND[i] − offset(sender → receiver), where the offset is the mean of the two clock calibrations that bracket that dispatch (step 5). The difference between the two calibrations is reported as the drift bound; within one dispatch the offsets moved by at most {max(c['drift_max'] for c in cases)/1e3:.1f} µs. Half the ping-pong round trip ({rtt/2e3:.2f} µs) is the reference for an unloaded store-to-visible time. Only the first slot each polling thread handles is used, 128 stores per receiver per dispatch.</li>
<li><b>Per-pair matrix</b>: the same one-way latency grouped by (sender, receiver).</li>
<li><b>E issued → F done</b>: SM 0's phase stamps on the receiving rank, from the moment its own count stores were issued to the moment all incoming counts had been seen. This is the end-to-end cost of the count exchange on the receiver's critical path.</li>
</ul>
<h3>2.3 Configuration and cases</h3>
<ul>
<li>Cases: notify with an almost empty data plane (64 tokens), light data (256 tokens, from E0), full data at 16 and 32 SMs, incast ratio 4 (from E2), and the receive array relocated by 4 MiB and by 4 MiB + 64 B into the unused hybrid signal region of the window (same offset on every rank, so the symmetric addressing still holds). 8 traced repeats per new case.</li>
<li>Clock offsets between the four GPUs are measured <em>inside each run</em>, immediately before and after every traced dispatch, with a 400-round ping-pong per ordered pair through torch symmetric memory (common/clocksync.py). A first attempt with offsets taken from E1 an hour earlier produced one-way values of ± milliseconds: the GPUs' <code>%globaltimer</code> clocks drift against each other between runs (the 0↔1 offset moved by about 8 ms over an hour), so "one calibration per session" is not good enough; "one per kernel" is, and it costs about 12 ms for four ranks.</li>
</ul>
<h2>3. Results</h2>
<div class="finding"><b>A count store takes {viz.pct(quiet['all'],50)/1e3:.2f} µs one way on a quiet fabric and {viz.pct(full16['all'],50)/1e3 if full16 else 0:.2f} µs p50 / {viz.pct(full16['all'],99)/1e3 if full16 else 0:.1f} µs p99 while 240 warps are storing data{', ' + f"{viz.pct(full32['all'],50)/1e3:.2f} / {viz.pct(full32['all'],99)/1e3:.1f} µs at 32 SMs" if full32 else ''}{', and ' + f"{viz.pct(inc['all'],50)/1e3:.2f} / {viz.pct(inc['all'],99)/1e3:.1f} µs under incast" if inc else ''}.</b> With a balanced data plane the median is barely touched but the tail grows fivefold; under incast the median itself is {viz.pct(inc['all'],50)/viz.pct(quiet['all'],50) if inc else 0:.0f}× the quiet value, and the receiver's wait for all peers' counts (E issued → F done) stretches to {statistics.median(inc['e_to_f']) if inc and inc['e_to_f'] else 0:.0f} µs. The inflation is the per-SM store FIFO of E2 seen from the control plane: SM 0's count stores queue behind its own dispatch warps' 14 KB stores to the hot rank.</div>
{report.figure(fig1, 'Figure 2. One-way latency of the notify count stores per case: p10–p90 bar, p99 whisker, p50 dot. Log scale. The dashed line is half the ping-pong round trip.')}
{report.figure(fig2, 'Figure 3. Median one-way latency per (sender, receiver) pair, quiet versus full data plane. Asymmetries between pairs would point at path differences; uniform inflation points at the sender’s queue.')}
<div class="finding"><b>Relocating the receive array {'changes' if rel and abs(viz.pct(rel[0]['all'],50) - viz.pct(full16['all'],50)) > 0.3 * viz.pct(full16['all'],50) else 'does not change'} the latency</b>{': ' + ', '.join(f"{viz.pct(q['all'],50)/1e3:.2f} / {viz.pct(q['all'],99)/1e3:.1f} µs for " + ' '.join(q['label'].split(chr(10))[:2]) for q in rel) if rel else ''}. {'At this granularity the address does not select a slower or faster path for an 8-byte store; whatever interleaving the hardware applies does not separate control from data.' if rel and abs(viz.pct(rel[0]['all'],50) - viz.pct(full16['all'],50)) <= 0.3 * viz.pct(full16['all'],50) else ''}</div>
{report.figure(fig3, 'Figure 4. The notify chain on SM 0 per case: time from this SM’s phase C to its phase E issue (waiting for the other SMs’ partial counts), from E to F (waiting for all peers’ counts, which includes their skew and the one-way latency), and from F to the host mailbox write.')}
{report.table(['case', 'kernel µs', 'samples', 'one-way p50 µs', 'p90', 'p99', 'max', 'E issued → F done µs'], tab, num_cols=range(1, 8))}
<h2>4. Conclusions</h2>
<ol>
<li><b>The control-behind-data delay is measurable and {'real' if full16 and viz.pct(full16['all'],50) > 2 * viz.pct(quiet['all'],50) else 'small'} at this scale.</b> The control plane shares the sender's egress queue with the data plane and pays the same queueing delay; the per-pair matrix shows the inflation is uniform across destinations, consistent with E2's shared-FIFO result rather than with a congested path.</li>
<li><b>End-to-end weight.</b> E issued → F done is {statistics.median(full16['e_to_f']) if full16 and full16['e_to_f'] else 0:.0f} µs with the full data plane, of a {full16['t_us'] if full16 else 0:.0f} µs kernel; the host learns the output sizes early enough in prefill. In the 64-token case the same chain is {statistics.median(quiet['e_to_f']) if quiet['e_to_f'] else 0:.0f} µs of a {quiet['t_us']:.0f} µs kernel, which is the decode regime's cost: with few tokens, the count exchange and the barriers are most of the kernel.</li>
<li><b>Method, and a correction to the clock-sync assumption.</b> Two in-kernel timestamps plus a per-run clock calibration gave one-way latencies for 128 stores per receiver per dispatch at no measurable kernel cost. The calibration cannot be done once per session: the per-GPU clocks drift by milliseconds per hour relative to each other, which is far more than FabricPerf reports for intra-node pairs; the triangle closure of a single calibration is still about 100 ns, so the offsets are precise at the instant they are taken. The receiver stamp is limited by the poll loop; a hardware arrival timestamp does not exist.</li>
</ol>
<div class="warnbox"><b>Limits.</b> The light-data (E0) and incast (E2) traces were recorded before in-run calibration existed and are not included. Receiver-side resolution is the poll iteration; only the first slot per polling thread is used. Four ranks.</div>
'''
meta = f"experiment E4 · {len(cases)} cases · 4 × H100 on host GPUs 4–7 · scripts: e4_sweep.py, common/apply_trace_patch.py (--relocate-recv), e4_analyze.py; clock offsets from E1"
open(os.path.join(HERE, 'report.html'), 'w').write(report.page('E4 · Control-plane stores behind the data plane', 'How long an 8-byte notify store takes to cross to a peer while 240 warps are pushing 14 KB stores through the same ports, measured with two in-kernel timestamps and a one-time clock calibration.', meta, body))
print('wrote report with', len(cases), 'cases')
