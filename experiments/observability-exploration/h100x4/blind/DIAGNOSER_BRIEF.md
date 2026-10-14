# Brief for the diagnoser

You are given one recorded DeepEP dispatch run that may or may not have been degraded by a hidden perturbation, plus a baseline run of the same configuration recorded without any perturbation. Your task is to decide whether the run is degraded, and if so, in which phase, at which location and from which cause, following the written method. Record every step you take and every number you compute, so that a reader can check the reasoning.

## What you may use

- `cases/<case>/run.pt` and `cases/<case>/nvlink.json` for your case, and `baseline/run.pt` and `baseline/nvlink.json`.
- The tooling in `../common/`: `traceload.py` (loader and phase helpers), `trace_layout.py` (record layout), `viz.py` and `report.py` (figures and HTML), `README.md` (how traces are made). Read `traceload.py` first.
- The method pages: `../../../../notes/deepep-diagnosis-method.html` (procedure, signal points, per-phase rules) and `../../../../notes/scaleup-comm-model.html` (the model behind it). The experiment reports under `../E*/report.html` show what normal values look like on this machine.

## What you must not do

- Do not read anything under `blind/sealed/`, `blind/attack.py`, `blind/noise/`, or any file that describes how the cases were made. Do not list processes or look at shell history. Do not run anything on the GPUs. The diagnosis is from the recorded data only.

## What the run contains

The run was recorded by `common/run_dispatch.py` on GPUs 4–7 (ranks 0–3), 16 SMs, 4,096 tokens per rank, with the per-lane trace on, host timestamps on, in-run clock calibration, and the Kineto kernel time (20 untraced iterations). `nvlink.json` holds the NVLink byte totals per GPU (summed over its 18 links) read once before the whole run started and once after it ended. Compare the case's totals with the baseline's totals: the workload is deterministic, so the baseline totals are the run's own traffic to the byte, and any excess in a case is traffic that did not come from this workload. Do not divide by a dispatch count to compare with a theoretical per-dispatch figure; the run also contains the timing iterations and their barriers.

## What to deliver

Write `cases/<case>/diagnosis.html` using `report.page` from `common/report.py`, with these sections: (1) what was compared with what, (2) the per-phase excess table, baseline vs case, per rank, (3) the tests you applied and their numbers, with at least two figures made with `viz.py`, (4) your conclusion: degraded or not; phase; location; cause; confidence (high / medium / low) and what evidence would raise it, (5) the alternatives you ruled out and why. Rasterise your figures with `common/render_check.py` and look at them before finishing. Keep the language plain: short sentences, define terms, no jargon without a one-line explanation.
