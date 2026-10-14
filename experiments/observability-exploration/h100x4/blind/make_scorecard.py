"""Scorecard: compares each diagnoser's conclusion (verdicts.json, filled in by the orchestrator after reading the diagnoses) with the sealed truth."""
import os, sys, json, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))
import report
mapping = json.load(open(os.path.join(HERE, 'sealed', 'mapping.json')))
verdicts = json.load(open(os.path.join(HERE, 'verdicts.json'))) if os.path.exists(os.path.join(HERE, 'verdicts.json')) else {}
rows = []; correct = 0; n = 0
for cid in sorted(mapping):
    truth = json.load(open(os.path.join(HERE, 'sealed', f'{mapping[cid]}.json')))
    v = verdicts.get(cid, {})
    ok = v.get('kind') == truth['kind']
    n += 1; correct += ok
    rows.append([f'<a href="cases/{cid}/diagnosis.html">{cid}</a>', truth['kind'], truth['desc'], v.get('degraded', '?'), v.get('phase', '?'), v.get('kind', '?'), v.get('detail', ''), v.get('confidence', '?'), '<span class="tag ok">correct</span>' if ok else ('<span class="tag warn">partly</span>' if v.get('partial') else '<span class="tag bad">wrong</span>'), v.get('note', '')])
body = f'''
<p class="lede">Eight recorded DeepEP dispatch runs, each with at most one hidden perturbation applied by an "attacker" script, were handed to a diagnoser that had not seen how they were made. The diagnoser had the scrubbed trace, a baseline, the NVLink byte totals of the whole run, the tooling, and the written method. This page compares its conclusions with the sealed truth.</p>
<h2>Results</h2>
<p>{correct} of {n} causes identified correctly.</p>
{report.table(['case', 'true cause', 'what was done', 'diagnosed as degraded?', 'phase found', 'cause found', 'detail', 'confidence', 'score', 'note'], rows)}
<h2>How the exercise was run</h2>
<ol>
<li>A baseline run (16 SMs, 4,096 tokens, 4 ranks on GPUs 4–7, trace on, host stamps on) was recorded with nothing else running.</li>
<li>For each case the attacker script applied one perturbation: an external process (a 4-process NCCL all-to-all job on the same GPUs; an HBM-copy or GEMM loop on one GPU) started 8 s before and stopped after the run, or a flag of the run driver (a host busy-wait or a GPU spin on one rank before each traced dispatch; a routing ratio; a different SM count). One case had no perturbation.</li>
<li>The recorded trace was scrubbed of every field that names the perturbation; the case order was shuffled and the mapping sealed in <code>blind/sealed/</code>.</li>
<li>One diagnoser per case, a fresh agent with the brief in <code>DIAGNOSER_BRIEF.md</code>, wrote <code>cases/&lt;case&gt;/diagnosis.html</code> from the data alone.</li>
<li>The orchestrator read each diagnosis, extracted its verdict into <code>verdicts.json</code> without changing it, and this page was generated.</li>
</ol>
'''
html = report.page('Blind diagnosis scorecard', 'Hidden perturbations of DeepEP dispatch, diagnosed from the recorded data alone, compared with the sealed truth.', f'blind cases · h100x4 · {time.strftime("%Y-%m-%d")}', body)
open(os.path.join(HERE, 'scorecard.html'), 'w').write(html); print('wrote scorecard.html', correct, '/', n)
