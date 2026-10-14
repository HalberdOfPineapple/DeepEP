"""Write index.html for the observability-exploration series."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'common'))
import report

E4 = open(os.path.join(HERE, 'E4-control-behind-data', 'summary.txt')).read().strip() if os.path.exists(os.path.join(HERE, 'E4-control-behind-data', 'summary.txt')) else 'Report pending.'

overview_svg = '''<svg viewBox="0 0 880 400" role="img" aria-label="The ten experiments placed on the phases of one dispatch: E6 and E9 on sync-in and the host path, E4 and E9 on the count exchange, E0, E3, E5 and E7 on the per-token loop and the sender SM, E1, E2 and E8 on the transfer, E0, E6 and E7 on sync-out">
<text x="20" y="24" class="head">Where each experiment looked (phases of one dispatch, as in the analysis note's frame)</text>
<rect x="20" y="40" width="120" height="44" rx="8" class="box-ok"/><text x="80" y="58" text-anchor="middle" class="head">A · sync in</text><text x="80" y="76" text-anchor="middle" class="small">launch, barrier</text>
<rect x="150" y="40" width="120" height="44" rx="8" class="box-ok"/><text x="210" y="58" text-anchor="middle" class="head">B · counts</text><text x="210" y="76" text-anchor="middle" class="small">8 B stores</text>
<rect x="280" y="40" width="220" height="44" rx="8" class="box-ok"/><text x="390" y="58" text-anchor="middle" class="head">C · per-token loop in the SM</text><text x="390" y="76" text-anchor="middle" class="small">slot wait · prepare · issue</text>
<rect x="510" y="40" width="150" height="44" rx="8" class="box-bad"/><text x="585" y="58" text-anchor="middle" class="head">D · transfer</text><text x="585" y="76" text-anchor="middle" class="small">queue → link → peer</text>
<rect x="670" y="40" width="110" height="44" rx="8" class="box-ok"/><text x="725" y="58" text-anchor="middle" class="head">E · sync out</text><text x="725" y="76" text-anchor="middle" class="small">tail, barrier</text>
<rect x="790" y="40" width="75" height="44" rx="8" class="box-ok"/><text x="827" y="58" text-anchor="middle" class="head">F · host</text><text x="827" y="76" text-anchor="middle" class="small">return</text>
<g class="small">
<rect x="20" y="100" width="120" height="34" rx="6" class="box"/><text x="80" y="121" text-anchor="middle">E6 idle: delay − 85 µs</text>
<rect x="20" y="140" width="120" height="34" rx="6" class="box"/><text x="80" y="161" text-anchor="middle">E5 launch waits 1–2 ms</text>
<rect x="20" y="180" width="120" height="34" rx="6" class="box"/><text x="80" y="201" text-anchor="middle">E9 call→entry 63 µs</text>
<rect x="150" y="100" width="120" height="34" rx="6" class="box"/><text x="210" y="121" text-anchor="middle">E4 count 1.8 → 70 µs</text>
<rect x="150" y="140" width="120" height="34" rx="6" class="box"/><text x="210" y="161" text-anchor="middle">E9 counts 6 µs of 142</text>
<rect x="280" y="100" width="220" height="34" rx="6" class="box"/><text x="390" y="121" text-anchor="middle">E0 issue stall = half the loop; 33 GB/s per SM</text>
<rect x="280" y="140" width="220" height="34" rx="6" class="box"/><text x="390" y="161" text-anchor="middle">E3 slot atomics 96 ns, padding irrelevant</text>
<rect x="280" y="180" width="220" height="34" rx="6" class="box"/><text x="390" y="201" text-anchor="middle">E5 HBM neighbour: load ×1.6, issue ×1.3</text>
<rect x="280" y="220" width="220" height="34" rx="6" class="box-acc"/><text x="390" y="241" text-anchor="middle">E7 finish = physical TPC, 1 : 1.2 : 1.4</text>
<rect x="510" y="100" width="150" height="34" rx="6" class="box"/><text x="585" y="121" text-anchor="middle">E1 wait = local acceptance</text>
<rect x="510" y="140" width="150" height="34" rx="6" class="box"/><text x="585" y="161" text-anchor="middle">E2 incast via sender FIFO</text>
<rect x="510" y="180" width="150" height="34" rx="6" class="box"/><text x="585" y="201" text-anchor="middle">E8 NCCL beside: ×2</text>
<rect x="510" y="220" width="150" height="34" rx="6" class="box-warn"/><text x="585" y="241" text-anchor="middle">E8 NVML read costs 20 %</text>
<rect x="670" y="100" width="110" height="34" rx="6" class="box"/><text x="725" y="121" text-anchor="middle">E0 tail 240 µs</text>
<rect x="670" y="140" width="110" height="34" rx="6" class="box"/><text x="725" y="161" text-anchor="middle">E7 tail = TPC levels</text>
<rect x="670" y="180" width="110" height="34" rx="6" class="box"/><text x="725" y="201" text-anchor="middle">E6 peers idle, 1:1</text>
<rect x="790" y="100" width="75" height="34" rx="6" class="box"/><text x="827" y="121" text-anchor="middle">E9 37 µs</text>
</g>
<text x="20" y="290" class="head">Three results that change the picture</text>
<text x="20" y="312" class="small">1. The per-token wait is local acceptance (E1): the sender has no per-token delivery signal; what limits a warp is the SM's ~7-store queue (E0, E2).</text>
<text x="20" y="330" class="small">2. The tail is set by which physical TPCs the launch lands on, stable per GPU, not by the warps' work (E7): a location inside the dark SM-to-port path.</text>
<text x="20" y="348" class="small">3. Outside the kernel matters as much as inside: 85 µs launch allowance (E6), 63 + 37 µs host path in decode (E9), 1–2 ms launch wait beside</text>
<text x="20" y="366" class="small">   compute (E5), and the observer itself, one NVML read, costing 20 % (E8).</text>
<text x="20" y="390" class="small">GPUs 4–7 of the 8×H100 node, 4 ranks, DeepEP V2.5 @ 93eb6eb, 2026-10-12/13. The other job on GPUs 0–3 did not use NVLink.</text>
</svg>'''

body = f'''
<p>Ten experiments from the programme in <a href="../../../notes/deepep-observability-analysis.html">notes/deepep-observability-analysis.html</a>, run on GPUs 4–7 of this node. Each folder holds the scripts, the raw results (<code>results/*.pt</code> or <code>.json</code>), the logs, and a self-contained <code>report.html</code> with motivation, method, results and diagrams. <a href="common/README.md">common/README.md</a> documents the tooling: a private copy of the installed DeepEP package with an instrumented dispatch kernel, a per-lane trace smuggled out through an existing tensor pointer, and in-run cross-GPU clock calibration.</p>
{report.figure(overview_svg, 'Figure 1. The ten experiments placed on the phases of one dispatch, with each one’s headline number.')}
<h2>Reports</h2>
<table><thead><tr><th>#</th><th>Question</th><th>Answer in one line</th><th>Report</th></tr></thead><tbody>
<tr><td>E1</td><td>What does <code>cp.async.bulk.wait_group</code> complete on, when the destination is a peer GPU? How far can one SM run ahead of the link?</td><td>It completes on local acceptance (150–400 ns, identical for local and peer); remote visibility needs a system-scope release (~1.8 µs ≈ RTT). The per-lane queue admits ~7 stores and drains at ~50 GB/s per SM; back-pressure lands on the store instruction. Data-then-flag with only <code>wait_group</code> gives stale reads.</td><td><a href="E1-bulk-store-completion/report.html">E1 report</a></td></tr>
<tr><td>E0</td><td>With the per-token wait being local, what bounds dispatch throughput?</td><td>Per-SM bulk-store capacity (~33 GB/s remote per SM) up to a per-GPU plateau of ~285 GB/s; half of a warp's 17 µs iteration is store-issue stall even on an idle fabric; link saturation shows as slot-wait tails of 36 µs; the warp tail is ~240 µs of a 540 µs data plane. The note's "window closes at 45 µs" is withdrawn.</td><td><a href="E0-window-vs-links/report.html">E0 report</a></td></tr>
<tr><td>E2</td><td>Does a hot destination slow a sender's stores to idle destinations?</td><td>Yes, through the sender's own per-SM FIFO: per-destination latency is identical for hot and idle destinations at every ratio, the hot rank's intake caps near 320 GB/s, and idle destinations' stores stretch over the whole kernel instead of finishing early. Releasing the slot earlier (<code>wait_group.read</code>) changes nothing.</td><td><a href="E2-incast-victim-flows/report.html">E2 report</a></td></tr>
<tr><td>E3</td><td>Do ~15,000 returning atomics on one 32-byte sector cost anything?</td><td>No: 96 ns p50 and ≤128 ns p99 with or without padding, kernel time unchanged. A non-problem at four ranks; worth repeating at eight and beyond.</td><td><a href="E3-address-hotspots/report.html">E3 report</a></td></tr>
<tr><td>E4</td><td>How long does an 8-byte notify store take to cross while 240 warps push 14 KB stores through the same ports?</td><td>{E4}</td><td><a href="E4-control-behind-data/report.html">E4 report</a></td></tr>
<tr><td>E5</td><td>When the GPU also runs compute, which step of the dispatch pays, and on which side?</td><td>HBM-bound neighbours stretch the token load (×1.6) and the store path (×1.3) on the loaded sender; a loaded <em>receiver</em> does not slow the stores into it. GEMMs cost little. The largest effect is the cooperative launch waiting 1–2 ms for free SMs, invisible inside the kernel.</td><td><a href="E5-compute-interference/report.html">E5 report</a></td></tr>
<tr><td>E6</td><td>If one rank is late, can the stamps say who, and by how much?</td><td>Yes: entry/arrive/release stamps on one clock recover the delay and name the rank. A GPU-side delay under ~85 µs is hidden behind the host launch path; beyond that the peers idle one for one. Host stamps tell a busy GPU from a slow host.</td><td><a href="E6-skew-injection/report.html">E6 report</a></td></tr>
<tr><td>E7</td><td>Is the 240 µs warp tail scheduling, routing or path?</td><td>Path, and a specific one: a warp’s finish is set by the physical TPC its block landed on (R² 88 %), in two or three fixed levels per GPU (1.0 : 1.2 : 1.4); tokens, stores and hot-rank stores explain ~0 %. Placement is identical run after run.</td><td><a href="E7-tail-attribution/report.html">E7 report</a> · <a href="E7-tail-attribution/one-iteration.html">one iteration, one warp</a></td></tr>
<tr><td>E8</td><td>What do two kernels on the same links cost each other, and who can tell?</td><td>Dispatch beside combine: both within 10 %. Dispatch beside a NCCL all-to-all: data plane ×2, with the same stamp signature as self-saturation. Per-link counters attribute it but perturb the kernel: one NVML read before launch cost 20 % and 300 µs of launch stagger.</td><td><a href="E8-shared-links/report.html">E8 report</a></td></tr>
<tr><td>E9</td><td>What is the decode critical path made of?</td><td>142 µs from call to return at 128 tokens: 63 µs host launch path, 19 µs opening barrier (18 µs entry skew), 6 µs counts, 13 µs data, 5 µs closing barrier, 37 µs epilogue and host return. The kernel is 43 µs; Kineto reports 32. Padding, SM count and CPU sync move only the segments they touch.</td><td><a href="E9-decode-critical-path/report.html">E9 report</a></td></tr>
</tbody></table>
<h2>What changed in the analysis after measuring</h2>
<ul>
<li><b>Sender-side per-token delivery is unobservable (note challenge C2):</b> the sender's only per-token event reports local acceptance; the sensor for the peer is a system-scope release, which costs a round trip. Receiver-side arrival stays unobservable in software.</li>
<li><b>Throughput bound rewritten (note problem P2):</b> the limit on how far a warp runs ahead is the per-SM async-proxy queue, not one token per warp per round trip; throughput is per-SM drain × SMs up to the per-GPU NVLink plateau; back-pressure appears first on the store instruction.</li>
<li><b>Incast victim flows sharpened (note problem P1):</b> the victim-flow effect exists and is produced upstream of the link, in the sender's shared FIFO; switch-level sharing is not needed to explain the data and is not excluded.</li>
<li><b>Slot-counter hotspot downgraded (note problem P3)</b> at four ranks; the measurement to repeat at scale is in E3.</li>
<li><b>Clock synchronisation:</b> a single ping-pong calibration closes triangles to ~100 ns, but the GPUs' <code>%globaltimer</code> clocks drift by milliseconds per hour against each other, so calibration must bracket each measured kernel (about 12 ms for four ranks). The note's "once per session" is withdrawn.</li>
<li><b>Tooling result:</b> a per-lane trace with 0.1–1 % overhead, written through an existing kernel argument, is enough for E0–E9; the library knows every store's destination, so address-classified accounting comes for free.</li>
<li><b>New after E5–E9 (phase attribution):</b> the frame's green phases (A, B, C, E, F) are decomposed end to end, including the host path, with one clock across ranks; the dark phase D gained one attributable location, the sender's TPC (E7), and one attributable cause, a foreign kernel on the links, visible only through counters outside the kernel (E8).</li>
<li><b>The observer is part of the system:</b> host-side counter reads (NVML) and cooperative-launch scheduling both perturb sub-millisecond kernels by more than the effects under study; a tool has to read from the GPU or place its reads where they have decayed.</li>
</ul>
<h2>Not done yet</h2>
<p>E10 (address-to-link mapping and link health; needs an address sweep per TPC, which E7 now motivates) and E11 (hybrid NVLink + RDMA pipeline; needs the second node). The eight-GPU repeats of E0, E2, E3 and E7 wait for GPUs 0–3. The cause of the per-TPC drain levels (E7) is the next question to settle, with a GPC map of these GPUs and an address sweep.</p>
'''
html = report.page('DeepEP observability exploration · h100x4', 'Experiments E0–E9 of the observability programme, run on GPUs 4–7 (four H100s): what each asked, what it found, and what it changed in the analysis.', f'experiments/observability-exploration · generated {time.strftime("%Y-%m-%d %H:%M")}', body)
open(os.path.join(HERE, 'index.html'), 'w').write(html); print('wrote index.html')
