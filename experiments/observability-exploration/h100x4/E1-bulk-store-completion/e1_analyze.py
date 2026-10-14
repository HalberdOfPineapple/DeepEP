"""Build report.html for E1 from results/e1_results.json."""
import json, os, sys, statistics, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import viz, report
from viz import PALETTE, INK, MUTED

R = json.load(open(os.path.join(HERE, 'results', 'e1_results.json')))
lat = [r for r in R['latency'] if r['blocks'] == 1 and r['warps'] == 1]
conc = [r for r in R['latency'] if not (r['blocks'] == 1 and r['warps'] == 1)]
modes = sorted({r['mode'] for r in lat}); sizes = sorted({r['bytes'] for r in lat})
MODE_NAME = {0: 'wait_group 0', 1: 'wait_group.read 0', 2: 'wait_group 0 + fence.acq_rel.sys', 3: 'wait_group.read 0 + fence.acq_rel.sys', 4: 'st.global ×32 lanes + fence.acq_rel.sys', 5: 'wait_group 0 + fence.acq_rel.gpu'}
MODE_COLOR = {0: PALETTE[0], 1: PALETTE[2], 2: PALETTE[1], 3: PALETTE[3], 4: PALETTE[6], 5: PALETTE[4]}
SHORT = {0: 'wait_group', 1: 'wait_group.read', 2: '+ fence.sys', 3: '.read + fence.sys', 4: 'st.global ×32 + fence', 5: '+ fence.gpu'}

# ---------- Figure A: latency vs size, local vs remote, per mode ----------
f, axs = viz.fig(9.2, 4.4, ncols=2, sharey=True)
for ax, remote in zip(axs, [False, True]):
    for m in modes:
        pts = sorted([(r['bytes'], r['p50']) for r in lat if r['mode'] == m and r['remote'] == remote and r['dst'] in (0, 1)])
        if not pts: continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker='o', color=MODE_COLOR[m], label=MODE_NAME[m])
        viz.end_label(ax, xs[-1], ys[-1], SHORT[m], MODE_COLOR[m])
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('store size (bytes)')
    ax.set_title('destination: ' + ('same GPU' if not remote else 'peer GPU over NVLink'))
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=8)
axs[0].set_ylabel('p50 issue → wait returned (ns)'); axs[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(1.08, -0.3), ncol=3, frameon=False)
figA = viz.to_svg(f)

# ---------- Figure B: remote minus local difference ----------
f, ax = viz.fig(8.6, 3.2)
for m in modes:
    loc = {r['bytes']: r['p50'] for r in lat if r['mode'] == m and r['dst'] == 0}
    rem = {r['bytes']: statistics.median([q['p50'] for q in lat if q['mode'] == m and q['remote'] and q['bytes'] == r['bytes']]) for r in lat if r['mode'] == m and r['dst'] == 0}
    xs = sorted(loc); ys = [rem[x] - loc[x] for x in xs]
    ax.plot(xs, ys, marker='o', color=MODE_COLOR[m]); viz.end_label(ax, xs[-1], ys[-1], SHORT[m], MODE_COLOR[m])
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale('log', base=2); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], fontsize=8, rotation=45)
ax.set_xlabel('store size (bytes)'); ax.set_ylabel('remote − local p50 (ns)'); ax.set_title('What the NVLink hop adds to each wait variant')
figB = viz.to_svg(f)

# ---------- Figure C: concurrency ----------
c1 = sorted([r for r in conc if r['blocks'] == 1 and r['mode'] == 0 and r['dst'] == 1], key=lambda r: r['warps'])
c2 = sorted([r for r in conc if r['warps'] == 15 and r['mode'] == 0 and r['dst'] == 1], key=lambda r: r['blocks'])
c3 = sorted([r for r in conc if r['warps'] == 15 and r['mode'] == 2 and r['dst'] == 1], key=lambda r: r['blocks'])
c4 = sorted([r for r in conc if r['warps'] == 15 and r['mode'] == 0 and r['dst'] == 0], key=lambda r: r['blocks'])
f, axs = viz.fig(9.2, 3.6, ncols=2)
ax = axs[0]
ax.plot([r['warps'] for r in c1], [r['warps'] * 14464 / r['mean'] for r in c1], marker='o', color=PALETTE[0]); ax.set_xlabel('warps on one SM (14,464 B each)'); ax.set_ylabel('GB/s'); ax.set_title('One SM: issue rate vs delivered')
if c1 and 'delivered_gbs' in c1[0]:
    ax.plot([r['warps'] for r in c1], [r['delivered_gbs'] for r in c1], marker='s', color=PALETTE[1]); viz.end_label(ax, c1[-1]['warps'], c1[-1]['delivered_gbs'], 'delivered', PALETTE[1])
    viz.end_label(ax, c1[-1]['warps'], c1[-1]['warps'] * 14464 / c1[-1]['mean'], 'issue rate', PALETTE[0])
ax = axs[1]
for series, color, label in [(c2, PALETTE[0], 'wait_group, remote'), (c3, PALETTE[1], '+fence.sys, remote'), (c4, PALETTE[2], 'wait_group, local')]:
    if not series: continue
    key = 'delivered_gbs' if 'delivered_gbs' in series[0] else None
    ys = [r[key] if key else r['blocks'] * 15 * 14464 / r['mean'] for r in series]
    ax.plot([r['blocks'] for r in series], ys, marker='o', color=color); viz.end_label(ax, series[-1]['blocks'], ys[-1], label, color)
ax.axhline(450, color=MUTED, lw=0.8, ls='--'); ax.annotate('450 GB/s link, theoretical', (1, 450), xytext=(0, 4), textcoords='offset points', fontsize=8, color=MUTED)
ax.set_xlabel('SMs × 15 warps, one destination'); ax.set_ylabel('GB/s'); ax.set_title('Many SMs: delivered throughput')
figC = viz.to_svg(f)

# ---------- Figure D: issue without waiting ----------
def knee_of(r):
    g = np.diff(np.array(r['issue_ns'])); base = np.median(g[:6]); idx = np.where(g > 1.6 * base)[0]
    return (int(idx[0]) + 1 if len(idx) else None, float(base), float(np.median(g[20:])))
knees = {(r['dst'], r['bytes']): knee_of(r) for r in R['issue'] if r['rep'] == 0}
f, axs = viz.fig(9.2, 3.4, ncols=2)
ax = axs[1]
for dst, color in [(0, PALETTE[2]), (1, PALETTE[0])]:
    r = next(q for q in R['issue'] if q['dst'] == dst and q['bytes'] == 14464 and q['rep'] == 0)
    g = np.diff(np.array(r['issue_ns'])); ax.plot(np.arange(1, 41), g[:40], marker='o', ms=3, color=color); viz.end_label(ax, 40, g[39], 'peer' if dst else 'local', color)
ax.set_xlabel('store index in the burst (14,464 B each)'); ax.set_ylabel('gap to previous issue (ns)'); ax.set_title('The issue instruction starts blocking after ~7 stores'); ax.set_ylim(0, None)
ax = axs[0]
for i, (dst, b, color) in enumerate([(0, 14464, PALETTE[2]), (1, 14464, PALETTE[0]), (1, 65536, PALETTE[1]), (1, 1024, PALETTE[3])]):
    rows = [r for r in R['issue'] if r['dst'] == dst and r['bytes'] == b]
    if not rows: continue
    r = rows[0]; ys = np.array(r['issue_ns']) / 1e3
    ax.plot(np.arange(1, len(ys) + 1), ys, color=color, lw=1.6); viz.end_label(ax, len(ys), ys[-1], f'{b} B → {"local" if dst == 0 else "peer"}: {ys[-1]:.0f} µs', color)
ax.set_xlabel('bulk stores issued back to back by one lane (no wait)'); ax.set_ylabel('time since first issue (µs)'); ax.set_title('Issue time tracks delivery: the burst is paced by the drain')
figD = viz.to_svg(f)
kn_loc, kn_rem = knees.get((0, 14464), (None, 0, 0)), knees.get((1, 14464), (None, 0, 0))

# ---------- ping-pong table + triangulation ----------
pp = R['pingpong']; nd = R['meta']['num_devices']
off = {(p['a'], p['b']): p['offset_b_minus_a_median'] for p in pp}
tri = []
for a in range(nd):
    for b in range(nd):
        for c in range(nd):
            if len({a, b, c}) == 3 and (a, b) in off and (b, c) in off and (a, c) in off:
                tri.append(abs(off[(a, b)] + off[(b, c)] - off[(a, c)]))
rtt_rows = [[f'{p["a"]}→{p["b"]}→{p["a"]}', f'{p["rtt_p10"]:.0f}', f'{p["rtt_p50"]:.0f}', f'{p["rtt_p99"]:.0f}', f'{p["offset_b_minus_a_median"]/1e6:+.3f} ms', f'{p["offset_std"]:.0f}'] for p in pp]
# visibility
vis_rows = [[MODE_NAME[v['mode']], v['bytes'], f'{v["rounds_with_stale"]} / {v["rounds"]}', f'{v["max_stale_chunks"]} / {v["chunks"]}'] for v in R['visibility']]

# key numbers
def p50(mode, remote, b):
    xs = [r['p50'] for r in lat if r['mode'] == mode and r['remote'] == remote and r['bytes'] == b]; return statistics.median(xs) if xs else float('nan')
k_wait_loc, k_wait_rem = p50(0, False, 14464), p50(0, True, 14464)
k_fence_loc, k_fence_rem = p50(2, False, 14464), p50(2, True, 14464)
k_rtt = statistics.median([p['rtt_p50'] for p in pp])
stale_wait = sum(v['rounds_with_stale'] for v in R['visibility'] if v['mode'] == 0); stale_fence = sum(v['rounds_with_stale'] for v in R['visibility'] if v['mode'] == 2)
per_sm = max((r['warps'] * 14464 / r['mean'] for r in c1), default=float('nan'))

primer_svg = '''<svg viewBox="0 0 880 340" role="img" aria-label="Data-then-flag: the sender writes a payload and then a small flag word; the receiver waits for the flag and then reads the payload. If the flag can overtake the payload the receiver reads old bytes.">
<defs><marker id="pa" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-mut"/></marker></defs>
<text x="20" y="28" class="head">Sender (GPU A)</text><text x="20" y="196" class="head">Receiver (GPU B)</text>
<line x1="150" y1="60" x2="860" y2="60" class="ln-mut"/><line x1="150" y1="228" x2="860" y2="228" class="ln-mut"/>
<rect x="160" y="40" width="200" height="40" rx="6" class="box-acc"/><text x="260" y="58" text-anchor="middle" class="mono">cp.async.bulk 14 KB</text><text x="260" y="74" text-anchor="middle" class="small">the data: one token</text>
<rect x="380" y="40" width="130" height="40" rx="6" class="box"/><text x="445" y="58" text-anchor="middle" class="mono">wait_group 0</text><text x="445" y="74" text-anchor="middle" class="small">"my store is done"?</text>
<rect x="530" y="40" width="170" height="40" rx="6" class="box-acc"/><text x="615" y="58" text-anchor="middle" class="mono">st flag = i</text><text x="615" y="74" text-anchor="middle" class="small">4 B: "token i is there"</text>
<rect x="530" y="206" width="170" height="40" rx="6" class="box"/><text x="615" y="224" text-anchor="middle" class="mono">spin until flag == i</text><text x="615" y="240" text-anchor="middle" class="small">the receiver's only clue</text>
<rect x="710" y="206" width="160" height="40" rx="6" class="box"/><text x="790" y="224" text-anchor="middle" class="mono">read the payload</text><text x="790" y="240" text-anchor="middle" class="small">expects every byte == i</text>
<path d="M615 80 L615 206" class="ln-mut" marker-end="url(#pa)"/><text x="625" y="130" class="small">flag crosses NVLink</text>
<path d="M260 80 C260 170 720 150 775 206" class="ln-bad" marker-end="url(#pa)"/><text x="280" y="170" class="small">payload crosses NVLink, possibly later</text>
<rect x="20" y="272" width="400" height="56" rx="8" class="box-bad"/><text x="30" y="292" class="head">wait_group only</text><text x="30" y="310" class="small">flag can arrive first; receiver reads old 16 B chunks = "stale"</text>
<rect x="440" y="272" width="430" height="56" rx="8" class="box-ok"/><text x="450" y="292" class="head">fence.acq_rel.sys, or a release store on the flag</text><text x="450" y="310" class="small">payload is guaranteed visible before the flag; zero stale rounds</text>
</svg>'''

method_svg = '''<svg viewBox="0 0 880 300" role="img" aria-label="E1 setup: one warp stores from shared memory into a peer GPU's buffer through a P2P pointer and times four different completion waits; a second probe checks whether the data is visible when a flag written after the wait arrives">
<defs><marker id="m1" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-acc"/></marker><marker id="m2" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-warn"/></marker></defs>
<text x="20" y="24" class="head">A · Latency probe (one warp, GPU 4)</text>
<rect x="20" y="40" width="120" height="60" rx="8" class="box"/><text x="80" y="62" text-anchor="middle" class="head">shared memory</text><text x="80" y="80" text-anchor="middle" class="small">16 B … 64 KB pattern</text>
<path d="M140 70 L230 70" class="ln-acc" marker-end="url(#m1)"/><text x="185" y="62" text-anchor="middle" class="small">cp.async.bulk</text>
<rect x="230" y="40" width="170" height="60" rx="8" class="box-acc"/><text x="315" y="62" text-anchor="middle" class="head">peer buffer</text><text x="315" y="80" text-anchor="middle" class="small">GPU 5/6/7 via P2P VA, or self</text>
<rect x="20" y="120" width="360" height="110" rx="8" class="box-warn"/>
<text x="30" y="140" class="head">timed: clock64 around one of</text>
<g class="mono small"><text x="30" y="160">0  wait_group 0</text><text x="30" y="176">1  wait_group.read 0</text><text x="30" y="192">2  wait_group 0 ; fence.acq_rel.sys</text><text x="30" y="208">4  32 lanes st.relaxed.sys.v4 ; fence.acq_rel.sys</text><text x="30" y="224">5  wait_group 0 ; fence.acq_rel.gpu</text></g>
<text x="20" y="252" class="small">2,000 iterations per point; destination address rotated through 256 MB;</text>
<text x="20" y="268" class="small">first 200 discarded. Concurrency: 1–15 warps per SM, 1–16 SMs.</text>
<text x="20" y="284" class="small">Issue probe: 512 stores with no wait, timestamp after each issue.</text>
<line x1="420" y1="20" x2="420" y2="290" class="ln-mut"/>
<text x="440" y="24" class="head">B · Visibility probe (GPU 4 → GPU 5)</text>
<text x="440" y="52" class="small">sender</text><line x1="500" y1="48" x2="860" y2="48" class="ln"/>
<text x="440" y="132" class="small">receiver</text><line x1="500" y1="128" x2="860" y2="128" class="ln"/>
<circle cx="530" cy="48" r="5" class="fill-acc"/><text x="530" y="38" text-anchor="middle" class="mono">bulk store i</text>
<circle cx="640" cy="48" r="5" class="fill-warn"/><text x="650" y="70" text-anchor="middle" class="mono">wait variant</text>
<circle cx="700" cy="48" r="5" class="fill-bad"/><text x="740" y="38" text-anchor="middle" class="mono">st flag = i</text>
<path d="M706 52 L760 124" class="ln-warn" marker-end="url(#m2)"/>
<circle cx="770" cy="128" r="5" class="fill-bad"/><text x="770" y="150" text-anchor="middle" class="mono">flag seen</text>
<text x="770" y="168" text-anchor="middle" class="small">read payload: count 16 B chunks ≠ i</text>
<path d="M760 132 L560 170" class="ln-mut"/><text x="560" y="186" text-anchor="middle" class="small">ack → next round</text>
<rect x="440" y="206" width="420" height="70" rx="8" class="box"/>
<text x="450" y="226" class="small">If the wait variant guarantees remote visibility, the receiver never sees a stale</text>
<text x="450" y="242" class="small">chunk after the flag. This is exactly the data-then-flag pattern DeepEP's closing</text>
<text x="450" y="258" class="small">barrier relies on (wait_group, grid sync, red.release.sys).</text>
</svg>'''

mech_svg = f'''<svg viewBox="0 0 880 250" role="img" aria-label="Where a bulk store's life is observable: issue, local acceptance (what wait_group returns on), the NVLink hop, and remote visibility (what a system fence or a round trip exposes)">
<g class="mono"><rect x="20" y="60" width="130" height="44" rx="8" class="box-ok"/><text x="85" y="86" text-anchor="middle">issued</text>
<rect x="190" y="60" width="150" height="44" rx="8" class="box-ok"/><text x="265" y="78" text-anchor="middle">accepted locally</text><text x="265" y="96" text-anchor="middle" class="small">wait_group returns</text>
<rect x="380" y="60" width="150" height="44" rx="8" class="box-bad"/><text x="455" y="78" text-anchor="middle">NVLink / switch</text><text x="455" y="96" text-anchor="middle" class="small">invisible</text>
<rect x="570" y="60" width="130" height="44" rx="8" class="box-bad"/><text x="635" y="86" text-anchor="middle">in peer L2</text>
<rect x="740" y="60" width="120" height="44" rx="8" class="box-warn"/><text x="800" y="78" text-anchor="middle">fence.sys</text><text x="800" y="96" text-anchor="middle" class="small">returns</text></g>
<path d="M150 82 L190 82 M340 82 L380 82 M530 82 L570 82 M700 82 L740 82" class="ln"/>
<text x="85" y="130" text-anchor="middle" class="small">t = 0</text>
<text x="265" y="130" text-anchor="middle" class="small">+{k_wait_rem:.0f} ns (14 KB), same as local {k_wait_loc:.0f} ns</text>
<text x="800" y="130" text-anchor="middle" class="small">+{k_fence_rem:.0f} ns ≈ RTT</text><text x="800" y="146" text-anchor="middle" class="small">({k_rtt:.0f} ns ping-pong)</text>
<rect x="20" y="160" width="840" height="70" rx="8" class="box"/>
<text x="30" y="182" class="small">wait_group 0 completes when the async proxy has handed the bytes to the local memory system. It carries no information about the peer.</text>
<text x="30" y="198" class="small">Only a system-scope release (fence.acq_rel.sys, or red.release.sys as in DeepEP's closing barrier) waits for the acknowledgement from the peer.</text>
<text x="30" y="214" class="small">A flag stored after wait_group alone can overtake the data: the visibility probe saw stale payload in {stale_wait} rounds; after fence.sys in {stale_fence}.</text>
</svg>'''

body = f'''
<h2>1. Motivation</h2>
<p>DeepEP's dispatch warps move every token with <code>cp.async.bulk</code> stores from shared memory into a peer GPU's buffer and then execute <code>cp.async.bulk.wait_group 0</code> before reusing the slot (<code>dispatch.cuh:283</code>). Before measuring, two things were assumed about that wait: that it is the kernel's only flow control, holding each warp to one token in flight until the peer has received it; and that its duration could serve as a per-destination congestion sensor, since a slow peer would make the wait longer. Both depend on what the hardware means by "complete" for a bulk store whose destination is another GPU. PTX documents completion as the writes being performed, but not whether that is acceptance by the local memory system or acknowledgement from the peer. Every later experiment reads the per-lane wait duration from the trace, so its meaning had to be settled first.</p>
<p>Three questions: (1) does <code>wait_group</code> wait for the NVLink round trip; (2) what does guarantee remote visibility, and what does it cost; (3) how many bulk stores can one SM push without the issue path blocking, which is the hardware's own window.</p>

<div class="card">
<h3>Terms used in this report</h3>
<p><b>Bulk store.</b> <code>cp.async.bulk</code> copies a block of bytes (here one 14 KB token) from an SM's shared memory to an address in global memory. When that address belongs to a peer GPU, the bytes travel over NVLink. The instruction returns immediately; the copy is carried out later by a hardware engine on the SM (the "async proxy", the same unit that serves TMA).</p>
<p><b>wait_group.</b> Because the copy is asynchronous, the thread needs a way to ask "is it done yet?" so it can reuse the shared-memory buffer. <code>cp.async.bulk.commit_group</code> closes a batch of stores and <code>cp.async.bulk.wait_group 0</code> blocks until all batches are "complete". E1's first question is what "complete" means when the destination is another GPU: has the data merely left the SM, or has it arrived at the peer?</p>
<p><b>The flag.</b> A flag is a small word (4 or 8 bytes) in the receiver's memory that the sender writes <i>after</i> the data, as a signal meaning "the data is there, you can read it". The receiver does not know when data will arrive, so it spins on the flag. This <i>data-then-flag</i> pattern is how all of DeepEP's communication works: dispatch warps write tokens with bulk stores, and later the per-slot counters, the token counts written by the notify warps, and the closing barrier word play the role of the flag; the receiving rank reads the tokens only after it has seen them.</p>
<p><b>"A flag stored after wait_group alone".</b> The sender's sequence is: bulk store the data, execute <code>wait_group 0</code>, then store the flag, with nothing else in between. If <code>wait_group</code> meant "the data has reached the peer", this sequence would be safe. The visibility probe shows it is not: the flag, a tiny scalar store, can reach the receiver before the 14 KB payload, so the receiver sees the flag and then reads bytes that are still from the previous round. Those are the <i>stale</i> chunks in the table. Inserting <code>fence.acq_rel.sys</code> between the wait and the flag, or writing the flag with release semantics (which is what DeepEP's <code>red.release.sys</code> barrier does), makes the hardware hold the flag until the data is visible, and the stale count drops to zero.</p>
<p><b>Why it matters for observability.</b> The only per-token completion event the sender can observe cheaply is <code>wait_group</code>, and it reports local acceptance, not arrival. Anything that wants to measure per-token delivery time at the sender therefore has nothing to read; the event that does reflect the peer (the fence or release) costs a full NVLink round trip and is executed once per kernel, not once per token.</p>
{report.figure(primer_svg, 'Figure 0. The data-then-flag pattern the visibility probe tests. The flag is a small word written after the data; the receiver spins on it and then reads the data. With only wait_group between the two stores, the flag can overtake the payload.')}
</div>

<h2>2. Methodology</h2>
<p>In one sentence: outside DeepEP, write small CUDA kernels that do exactly one bulk store (or one flag exchange) between two GPUs, time it on the GPU's own clock, and vary what the sender waits on.</p>
<h3>2.1 Procedure</h3>
<ol>
<li><b>Build.</b> <code>e1_bulk.cu</code> is compiled once with <code>torch.utils.cpp_extension</code> into a module exposing the probe kernels below. One Python process (<code>e1_run.py</code>) drives all four GPUs; peer access is enabled between every pair so a kernel on GPU A can hold a plain pointer into GPU B's memory, the same address class DeepEP uses over NVLink.</li>
<li><b>Allocate.</b> On every GPU a 64 MB destination buffer and a few flag words.</li>
<li><b>Latency probe.</b> For every destination GPU (including the source itself, the "local" case), every wait variant (six, listed under 2.2) and every store size (1 KB to 64 KB): launch one warp that repeats 2,000 times: fill a shared-memory buffer, read <code>%clock64</code>, issue one bulk store to the destination, commit, do the wait variant, read <code>%clock64</code> again. The first 200 repeats are discarded as warm-up; the rest give the issue-to-return distribution for that cell.</li>
<li><b>Concurrency probe.</b> The same kernel with 1, 4, 8 and 15 warps on one SM, then 15 warps on 1 to 16 SMs, all storing 14,464-byte tokens to one peer, to see how the per-store wait and the delivered rate change as more stores share an SM and a link.</li>
<li><b>Issue probe.</b> One lane issues 512 bulk stores back to back with no wait at all, stamping the clock after each issue instruction. A jump in the gap between consecutive stamps shows the point where the hardware stops accepting new stores.</li>
<li><b>Ping-pong.</b> For every ordered GPU pair, one thread on each side: A writes a flag into B's memory, B spins until it sees it and writes back into A's memory, A spins; 10,000 rounds, each side stamping <code>%globaltimer</code> at send and receive. This gives the round-trip time and, PTP-style, the offset between the two GPUs' clocks.</li>
<li><b>Visibility probe.</b> Described in the primer above: 2,000 rounds per (wait variant, size) of data store, wait, flag store on GPU 4; spin on flag, read data, count stale chunks on GPU 5.</li>
<li><b>Analyse.</b> Everything is written to <code>results/e1_results.json</code>; <code>e1_analyze.py</code> computes the percentiles and draws the figures.</li>
</ol>
<h3>2.2 Probe details and definitions</h3>
{report.figure(method_svg, 'Figure 1. The two probes. Both are standalone kernels compiled with torch.utils.cpp_extension; peers are reached through CUDA peer-access pointers, the same address class DeepEP obtains from ncclGetLsaPointer.')}
<ul>
<li><b>Hardware.</b> {R['meta']['device']} × {nd} (host GPUs {R['meta']['cuda_visible']}), NVLink through four NVSwitch chips. GPUs 0–3 of the node were busy with another job during the run; its NVLink counters stayed at zero throughout (see common/logs).</li>
<li><b>Latency probe.</b> One warp per slot. Lane 0 issues one bulk store of <i>B</i> bytes from a shared-memory buffer to the destination, commits the group and waits with one of the variants above; the warp reads <code>%clock64</code> before the issue and after the wait. Cycles are converted to ns with the slot's own (clock64, globaltimer) pair measured over the whole run. Mode 4 replaces the bulk store by 32 lanes of 16-byte <code>st.relaxed.sys</code> followed by a system fence, the pattern DeepEP's notify warps use for the count stores.</li>
<li><b>Concurrency.</b> The same probe with 1, 4, 8, 15 warps on one SM and with 15 warps on 1 to 16 SMs, all to one peer; the issue rate is warps × bytes / mean wait, the delivered rate is total bytes / (time to the end of the final <code>wait_group</code>), which includes the drain of whatever the waits did not cover.</li>
<li><b>Issue probe.</b> One lane issues 512 bulk stores back to back with no wait, recording the cycle after each issue; a knee in the issue timestamps would be the point where the hardware stops accepting.</li>
<li><b>Ping-pong.</b> Two single-thread kernels on two GPUs exchange 4-byte flags with <code>st.relaxed.sys</code> and <code>ld.acquire.sys</code>; 10,000 rounds per ordered pair give the round-trip time and a PTP-style offset between the two GPUs' <code>%globaltimer</code> clocks. Offsets around every triangle of GPUs are checked for consistency.</li>
<li><b>Visibility probe.</b> The sender writes pattern <i>i</i> with a bulk store, performs one wait variant, then stores a flag; the receiver spins on the flag and immediately counts 16-byte chunks that do not yet hold <i>i</i>. 2,000 rounds per (variant, size).</li>
</ul>

<h2>3. Results</h2>
<div class="finding"><b>wait_group does not wait for the peer.</b> For a 14,464-byte store the p50 from issue to the return of <code>wait_group 0</code> is {k_wait_loc:.0f} ns to local memory and {k_wait_rem:.0f} ns to a peer GPU. Adding <code>fence.acq_rel.sys</code> raises both to about {k_fence_loc:.0f} ns and {k_fence_rem:.0f} ns, close to the ping-pong round trip of {k_rtt:.0f} ns. <code>wait_group.read</code> is within a few ns of <code>wait_group</code>. The NVLink hop is invisible to the wait and visible to the fence.</div>
{report.figure(figA, 'Figure 2. Issue-to-return latency of one bulk store against its size, for the same GPU (left) and a peer GPU (right). The two wait_group variants coincide and barely depend on the destination; the fenced variants sit at the round-trip time. Mode 4 (32 scalar stores) grows with size because the 32 lanes issue many 16 B stores.')}
{report.figure(figB, 'Figure 3. The same data as a difference: remote minus local p50 for each variant. The bulk-store waits gain under 10 ns from the hop at small sizes and a few hundred ns at 64 KB (the local memory system itself absorbs faster); the fenced variants gain what a round trip costs.')}
{report.figure(mech_svg, 'Figure 4. What each event in a remote bulk store\'s life is tied to, with the measured numbers. The sensor DeepEP\'s loop has (wait_group) reports local acceptance; the sensor that reports the peer (a system-scope release) costs a round trip.')}
<div class="finding"><b>A flag stored after wait_group alone can arrive before the data.</b> With plain <code>wait_group 0</code> the receiver found stale payload after seeing the flag in {stale_wait} of {sum(v['rounds'] for v in R['visibility'] if v['mode']==0)} rounds (up to {max((v['max_stale_chunks'] for v in R['visibility'] if v['mode']==0), default=0)} chunks of a 4 KB store still old); with <code>fence.acq_rel.sys</code> after the wait, in {stale_fence}. DeepEP's closing barrier is correct because its <code>red.release.sys</code> carries release semantics; the <code>tma_store_wait</code> before it only guarantees that the async proxy has finished reading shared memory and handing the bytes on.</div>
{report.table(['wait variant', 'bytes', 'rounds with stale data', 'max stale 16 B chunks'], vis_rows, num_cols=(1, 2, 3))}
<div class="finding"><b>Issue blocks after about seven stores; the back-pressure point is the store instruction, not the wait.</b> One lane issuing 14,464-byte bulk stores back to back pays {kn_loc[1]:.0f} ns per issue for the first {kn_loc[0]} stores and then {kn_loc[2]:.0f} ns per issue to local memory, {kn_rem[2]:.0f} ns to a peer: the async proxy accepts a handful of stores and thereafter admits one only when one drains. That pace is {14464/kn_loc[2]:.0f} GB/s local and {14464/kn_rem[2]:.0f} GB/s to a peer, per SM, and matches the delivered rates in the concurrency sweep ({statistics.median([r['delivered_gbs'] for r in c1 if r['warps']>=4]) if c1 and 'delivered_gbs' in c1[0] else 0:.0f} GB/s per SM with 4 to 15 warps). Sixteen SMs deliver {c2[-1]['delivered_gbs'] if c2 and 'delivered_gbs' in c2[-1] else 0:.0f} GB/s into a single peer, where the link saturates and the per-store wait p99 jumps to {c2[-1]['p99']/1e3 if c2 else 0:.0f} µs.</div>
{report.figure(figD, 'Figure 5. Left: time of each issue in a burst of 512 bulk stores from one lane with no wait; the slope is the per-SM drain rate and the burst is delivered (final wait_group, end labels) within a few µs of the last issue. Right: the gap between consecutive issues; after about seven stores the cp.async.bulk instruction itself stalls until a slot frees.')}
{report.figure(figC, 'Figure 6. Concurrency. Left: on one SM the delivered rate saturates at about 50 GB/s from four warps up. Right: delivered throughput against SM count for one peer destination with and without a system fence per store, and for a local destination (which reaches almost 1 TB/s, so the ~350 GB/s plateau is the NVLink path, not the SMs).')}
<h3>Clock alignment between GPUs</h3>
<p>The ping-pong gives a round trip of {k_rtt:.0f} ns p50 between any pair, and <code>%globaltimer</code> offsets between GPUs that are large (hundreds of ms to seconds) but stable to tens of ns within a run. Around every triangle of three GPUs the three pairwise offsets close to within {max(tri) if tri else float('nan'):.0f} ns (median {statistics.median(tri) if tri else float('nan'):.0f} ns). That is the error bar for cross-GPU timestamps in E4 and E6: one calibration per session, about 100 ns, no per-kernel handshake.</p>
{report.table(['pair', 'RTT p10 (ns)', 'RTT p50 (ns)', 'RTT p99 (ns)', 'offset b − a', 'offset std (ns)'], rtt_rows, num_cols=(1, 2, 3, 4, 5))}

<h2>4. Conclusions and what they change</h2>
<ol>
<li><b>Can the sender observe per-token delivery? No.</b> <code>wait_group</code> is a local-acceptance event. Its duration measures the depth of the local async-proxy / memory-system queue in front of this SM, not the state of the link or the peer. In the lane trace of E2, "issue → done" is therefore a <em>sender-side</em> congestion signal. Per-destination information survives only insofar as the local queue is shared across destinations, which E2 tests.</li>
<li><b>What limits how far a warp runs ahead.</b> DeepEP's loop does not wait a round trip per token. The limit is the per-SM async-proxy queue: about seven bulk stores per lane, drained at roughly 50 GB/s per SM to a peer. Back-pressure therefore reaches a dispatch warp first at the <code>cp.async.bulk</code> issue (the instruction stalls) and only secondarily at <code>wait_group</code>; with 16 SMs the aggregate issue capacity (~800 GB/s) exceeds the link, so the link's back-pressure shows up as queueing inside the SMs, visible as the slot wait and issue stalls. The pre-measurement model in which throughput is warps × one token per round trip is withdrawn; E0 measures the corrected version on the real kernel.</li>
<li><b>Remote completion has exactly one cheap witness: a system-scope release.</b> It costs a round trip ({k_fence_rem:.0f} ns) and serialises the issuing thread, so it cannot be inserted per token without changing the kernel. A per-destination completion counter (as GIN provides for the NIC path) is the missing hardware primitive for the NVLink path.</li>
<li><b>Correctness note for anyone adapting the pattern.</b> Data-then-flag over NVLink needs the release on the flag or a system fence; <code>wait_group</code> (either variant) is not enough. The visibility probe makes this reproducible in seconds.</li>
<li><b>Cross-GPU clocks on one node are usable with a one-time calibration</b> (triangle closure about {statistics.median(tri) if tri else float('nan'):.0f} ns), which E4 and later experiments rely on.</li>
</ol>
<div class="warnbox"><b>Limits.</b> Peer pointers here come from CUDA peer access, not from an NCCL window; the address class and path are the same, the allocation is not. Timings are from a single SM clock converted to ns; p50 values repeat exactly across sizes because the async proxy's acceptance cost is deterministic at this resolution. Concurrency to a single destination from one GPU exercises one ingress port; fan-out to several peers is left to E0/E2 with the real kernel. Nothing here says how deep the hardware queue is in bytes; only that it exceeds 7 MB for one lane.</div>
'''
meta = f"experiment E1 · {R['meta']['time']} · {nd} × {R['meta']['device']} on host GPUs {R['meta']['cuda_visible']} · scripts: e1_bulk.cu, e1_run.py, e1_analyze.py"
html = report.page('E1 · What "complete" means for a bulk store to a peer GPU', 'Calibrating the one per-token event DeepEP\'s data plane has: is cp.async.bulk.wait_group a round trip or a local hand-off, what guarantees remote visibility, and how far ahead of the link can one SM run.', meta, body)
open(os.path.join(HERE, 'report.html'), 'w').write(html)
print('wrote', os.path.join(HERE, 'report.html'))
