# Common tooling for the observability-exploration series

Everything here runs against a **private copy** of the installed `deep_ep` package (`pkg/deep_ep`), selected with `PYTHONPATH=common/pkg`. The DeepEP repository and the conda installation are never modified. DeepEP's JIT compiles kernels from the headers inside the package directory, and the header contents are part of its cache key, so editing `pkg/deep_ep/include/...` is enough to get a new kernel.

## Files

| File | Purpose |
|---|---|
| `pkg/` | Copy of `site-packages/deep_ep` (version 2.5.0+93eb6eb). `include/deep_ep/impls/ep/dispatch.cuh` is the instrumented kernel. |
| `dispatch.cuh.pristine` | The unmodified upstream kernel the patch starts from. |
| `apply_trace_patch.py` | Generates the instrumented `dispatch.cuh`. Flags: `--wait-read` (use `cp.async.bulk.wait_group.read` in the loop), `--pad-counters` (one 128 B line per slot counter), `--relocate-recv N` (move the receive-count array N bytes into the unused hybrid signal region), `--no-trace`, `--no-sm / --no-notify / --no-loop` (probe groups), `--pristine` (restore). |
| `trace_layout.py` | Host-side layout of the trace region and the allocation helper. |
| `run_dispatch.py` | Runs a 4-rank dispatch (`torch.multiprocessing`), measures the untraced kernel time with Kineto (DeepEP's own method, L2 flush + all-reduce barrier per iteration), then records N traced repeats and saves everything to one `.pt`. |
| `one_shot.py` | Debug helper: a few traced dispatches and a printed SM-level timeline. |
| `viz.py`, `report.py` | Matplotlib-to-inline-SVG charts (reference palette, thin marks, direct labels) and the HTML report skeleton. |
| `traceload.py` | Loader that turns a saved run into per-rank, per-repeat records (SM events, warp records, token records, clock offsets) plus phase-duration and latency helpers; used by the E5–E9 analyses. |
| `clocksync.py` | In-run cross-GPU clock calibration (ping-pong through torch symmetric memory) bracketing every traced dispatch. |
| `hostclock.py`, `gtimer.cu` | Maps host `perf_counter_ns` onto the GPU `%globaltimer` (one-thread timer kernel bracketed by host stamps, ±5 µs) and provides a GPU-side spin for skew injection. |
| `nvml_links.py` | Reads per-link NVLink TX/RX byte counters (NVML field values) before and after a traced repeat. Note: one read of four GPUs takes tens of ms and perturbs the GPUs for about a millisecond afterwards (E8). |
| `sweep.py` | Shared sweep runner: list of (name, patch flags, run_dispatch args), skips existing results. |

## How the trace gets out of the kernel

The kernel signature is fixed by DeepEP's host code, so the trace buffer is smuggled through the optional `cumulative_local_expert_recv_stats` tensor: the host passes a 1-D view of length `num_local_experts` whose underlying allocation is much larger, and the kernel writes `int64` records after the first `num_local_experts` ints. A magic word in the header enables the probes; without it the instrumented kernel skips every probe (this is how the "untraced" timings are taken with the same binary). Cost with tracing on: about 0.1–1 % of kernel time.

Record layout (int64 indices relative to the trace base): header 0–15; SM events 16 + 32·sm + e (kernel entry, barrier arrive/release, notify phases); SEND 1040+i and RECV 3088+i (notify count stores, per slot); WARP 5136 + 4·warp; per-token records from 8192, 64 per token: `[0]` sm/warp, `[1..2]` slot-wait start/end, `[3]` atomic start, `[5]` load done, `[6]` stores issued, `[7]` destination mask, `[8+lane]` per-lane store issue, `[16+lane]` per-lane wait return, `[24+lane]` destination rank, `[32+lane]` slot index, `[40+lane]` per-lane atomic return.

All timestamps are `%globaltimer` (ns, 32 ns granularity). Cross-GPU alignment uses the offsets from E1's ping-pong (triangle closure ≈ 100 ns on this node).

## Running

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate deepep
export EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=$PWD/common/pkg
python common/apply_trace_patch.py            # baseline instrumented kernel
taskset -c 52-103 python common/run_dispatch.py --num-sms 16 --num-tokens 4096 --ratio 1 --out out.pt
```

`run_dispatch.py` options added for E5–E9: `--bg copy|gemm --bg-ranks all|0 --bg-iters N` (background kernels on a side stream), `--skew-rank R --skew-us X --skew-mode gpu|host` (delay one rank before the launch), `--host-ts 1` (host stamps mapped onto the GPU clock), `--cpu-sync 0|1`, `--overlap combine|nccl` (a second EPBuffer's combine, or NCCL all-to-all through a second process group, on a side stream), `--nvml 1` (per-link counters), `--no-kineto 1`.

`EP_DISABLE_GIN=1` is required on this node (no NIC, so NCCL GIN is unavailable); the test harness additionally needs `--allow-hybrid-mode 0`. Pin to socket-1 cores (52–103) so the other job on socket 0 does not add launch skew.

## Pitfalls met

- A warp-wide `__ballot_sync` inside a `lane_idx == 0` branch deadlocks the kernel silently (GPU at 100 %, no timeout, because the stall is not in a `timeout_while` loop). Probes that gather across lanes must run on all lanes.
- `bench_kineto` matches kernel names by substring; `dispatch` also matches `dispatch_copy_epilogue_impl`. Use `dispatch_impl<`.
- `pkill -f <pattern>` from a shell whose command line contains the pattern kills the shell.
- A cooperative launch (dispatch) waits until its SMs are free: a long kernel on another stream delays it by that kernel's whole duration, and even a queue of 30 µs kernels delays it by 1–2 ms (E5).
- The GPUs' `%globaltimer` clocks drift by milliseconds per hour against each other: calibrate inside each run (clocksync.py), never once per session (E4).
- `deep_ep` sets the default dtype to bfloat16 and the default device to CUDA: use integer arithmetic on timestamps and `device='cpu'` for pinned tensors.
- NVML field-value reads perturb the measured kernels for about a millisecond: never read counters between the barrier and the launch (E8).
