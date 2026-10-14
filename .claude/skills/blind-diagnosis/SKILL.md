---
name: blind-diagnosis
description: Run a blind attack-and-diagnose case study on DeepEP dispatch. One role (the attacker, a script or a subagent) degrades a run on GPUs 4–7 with one hidden perturbation and seals the truth; another role (the diagnoser, a fresh subagent) must find the phase, location and cause from the recorded data alone and write a detailed diagnosis report; the orchestrator scores the diagnoses against the sealed truth. Use when asked to validate the observability method blindly, to generate case studies, or to test a new signal or rule.
---

# Blind attack-and-diagnose case studies for DeepEP

## Purpose

The experiments E0–E9 under `experiments/observability-exploration/h100x4/` each knew their cause in advance. This skill tests the diagnosis method without that knowledge: the diagnoser sees a degraded run and a baseline, nothing else, and must name the owner of the slowdown. The result is a scorecard (which causes are recognised, which are confused with which) and, per case, a written diagnosis that records every step and number.

Read before starting: `notes/deepep-diagnosis-method.html` (the procedure the diagnoser must follow) and `notes/scaleup-comm-model.html` section 8 (the method behind it). Do not run anything on GPUs before checking `nvidia-smi`; GPUs 0–3 belong to another user; use GPUs 4–7 only.

## Files (all under `experiments/observability-exploration/h100x4/blind/`)

| File | Role |
|---|---|
| `attack.py` | The attacker. `--baseline` records the clean baseline; `--cases a,b,c` records those perturbations in shuffled order as `case01..`, scrubs the traces, writes `sealed/<case>.json` and `sealed/mapping.json`. The `ATTACKS` dict lists the available perturbations and is the place to add new ones. |
| `noise/noise_nccl.py` | External foreign NVLink traffic: 4 NCCL processes doing `all_to_all_single` until a stop file appears (own rendezvous port 8477). |
| `noise/noise_compute.py` | External compute neighbour on one GPU: 32 MB HBM copies or 2048³ GEMMs until a stop file appears. |
| `DIAGNOSER_BRIEF.md` | The rules and deliverable for the diagnoser. Give it verbatim. |
| `make_scorecard.py` | Builds `scorecard.html` from `sealed/*.json` and `verdicts.json`. |
| `cases/<case>/run.pt`, `nvlink.json` | What the diagnoser receives. `baseline/` holds the clean run in the same format. |

The underlying run driver is `common/run_dispatch.py`; its perturbation flags (`--skew-*`, `--bg`, `--overlap`, `--ratio`, `--num-sms`) are scrubbed from `run.pt` by `attack.py`. External perturbations leave no trace in the file at all except their effect.

## Available perturbations (keys of `ATTACKS`)

`none` (control) · `nccl` (foreign traffic, external) · `copy_gpu5` / `gemm_gpu6` (compute neighbour on one GPU, external) · `hostskew_r2` (host busy-wait 300 µs on rank 2) · `gpuskew_r3` (GPU spin 400 µs on rank 3) · `incast3` (routing ratio 3, visible in the routing itself) · `fewsms` (num_sms 4, visible in args: a misconfiguration case).

Each case takes about 2 minutes (warm-up, 20 Kineto iterations with the trace off, 5 traced repeats with clock calibration). Always include `none` so the diagnoser cannot assume a perturbation exists.

## Procedure

1. **Check the GPUs.** `nvidia-smi`; GPUs 4–7 must be idle. Ask the user which GPUs to use if anything differs from the usual.
2. **Record the baseline** (once per machine state):
   ```bash
   cd experiments/observability-exploration/h100x4/blind
   source /root/miniconda3/etc/profile.d/conda.sh && conda activate deepep
   python attack.py --baseline
   ```
3. **Record the cases** with a shuffled, sealed mapping:
   ```bash
   python attack.py --cases none,nccl,copy_gpu5,gemm_gpu6,hostskew_r2,gpuskew_r3,incast3,fewsms --shuffle-seed 7
   ```
   Run this in the background and monitor `common/logs/`. After it finishes, confirm `nvidia-smi` shows GPUs 4–7 idle and no `noise_*` process is left (`pgrep -af noise_`).
4. **Spawn one diagnoser per case** with the `Agent` tool (general-purpose, fresh context, never the same agent that saw the attack list). The prompt is: the full text of `DIAGNOSER_BRIEF.md`, the case name, and the absolute paths of `blind/`, `common/` and the two method notes. State explicitly that it must not read `sealed/`, `attack.py`, `noise/`, this skill, or the orchestrator's conversation. Diagnosers run CPU-only and can run in parallel. Expect 10–20 minutes each; they must produce `cases/<case>/diagnosis.html` with figures checked by `common/render_check.py`.
5. **Score without editing.** Read each `diagnosis.html`. Copy its verdict into `blind/verdicts.json` as `{"case01": {"degraded": "yes", "phase": "...", "kind": "<one of the ATTACKS kinds, or 'control'>", "detail": "...", "confidence": "...", "partial": false, "note": "..."}}`. Set `partial` when the phase and location are right but the cause is not. Do not change the diagnosis files. Then `python make_scorecard.py` and read `scorecard.html`.
6. **Report** to the user: the scorecard, the confusions (which pairs of causes were mixed up and which signal would have separated them), and anything the diagnoser needed that the data did not contain. Add confusions as new rules or new signals to the method note, and add them to this skill's "known confusions" list below.
7. **Publish** the scorecard and the diagnoses as supporting files of the experiments artifact (index at `h100x4/index.html`, URL in memory), after rasterising every figure and looking at it.

## Rules that keep the exercise honest

- The diagnoser never sees the attack list, the mapping, or the orchestrator's conversation. Give it files and the brief only.
- Scrubbing must remove every field that names the perturbation. If a new perturbation adds a flag to `run_dispatch.py`, add that flag to the `scrub()` list in `attack.py` before using it.
- Do not read NVML counters between the barrier and the launch inside the run (it perturbs the kernel by about 20 %); `attack.py` reads the link byte totals once before and once after the whole run, which is what the diagnoser gets.
- Keep `none` in every campaign, and consider including the same perturbation twice at different intensities.
- The baseline and the cases must share the configuration (ranks, tokens, SMs) except for the `fewsms` case, whose point is that the configuration itself is the cause.

## Adding a perturbation

Add an entry to `ATTACKS` in `attack.py`: `kind` (the class the scorecard matches on), `desc`, `flags` (run_dispatch flags) or `external` (`('nccl', MB)` or `('copy'|'gemm', gpu_index)`), optional `override` of base flags. If it needs a new driver flag, add the flag to `run_dispatch.py` and to `scrub()`. Candidates not yet implemented: a CPU hog pinned to one rank's cores (host contention without a flag), a second DeepEP job on the same GPUs, reduced SM clocks on one GPU (root-only, reversible; ask first), and a slow-TPC placement change via a different `num_sms`.

## Known confusions (fill in after each campaign)

- (none recorded yet; the first full campaign has not been scored). Harness verified on 2026-10-14 with `--baseline` and one external case (`copy_gpu5`): run.pt, nvlink.json, sealed truth and mapping all written; GPUs released cleanly.

## Expected signatures, for scoring sanity checks (from E0–E9)

- foreign traffic: data plane up to 2×, slot wait p99 15 → 55 µs, uniform over destinations and SMs; NVLink byte totals of the run far above the baseline's (the workload is deterministic: baseline and an unperturbed case agree to the megabyte, about 13.8 GB TX per GPU for the standard run).
- compute neighbour (HBM copies) on one GPU: on that rank C2 prepare ×1.6 and C3 issue ×1.3; its stores to peers slower; other ranks' stores into it unchanged; other ranks wait 1–2 ms at the opening barrier because that rank's cooperative launch is delayed.
- host skew on rank r: ranks ≠ r wait almost the full delay at the opening barrier; rank r's host call stamp is late while its call→entry stays about 65 µs.
- GPU skew on rank r: same wait on the others minus about 85 µs; rank r's call→entry grows by the delay while its host call is on time.
- routing incast: destination counts in `topk_idx`/`dst_counts` uneven; hot rank intake capped near 320 GB/s; per-destination store latency equal across destinations.
- fewer SMs: visible in `args`; data plane scales with 1/SMs; per-SM rates unchanged.
- control: every phase within baseline noise (a few percent); the per-TPC finish levels identical to the baseline.
