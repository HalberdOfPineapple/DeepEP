"""E3: address hot spots. Baseline vs padded slot counters, at several concurrencies and token counts.

The barrier-word variant (one word per sender) is not implemented: the NVLink barrier's phase/sign scheme would have to
change on both the arrive and the wait side; the per-rank arrive→release time is reported from the trace instead.
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
env = dict(os.environ, EP_DISABLE_GIN='1', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7'), PYTHONPATH=os.path.join(ROOT, 'common', 'pkg'))
configs = [(16, 4096), (32, 4096), (16, 256), (32, 256)]
for vname, vflags in [('base', []), ('padded', ['--pad-counters'])]:
    subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')] + vflags, check=True)
    for sms, tok in configs:
        out = os.path.join(HERE, 'results', f'{vname}_sms{sms}_tok{tok}.pt')
        if os.path.exists(out): print('skip', out); continue
        cmd = ['taskset', '-c', '52-103', sys.executable, os.path.join(ROOT, 'common', 'run_dispatch.py'), '--num-sms', str(sms), '--num-tokens', str(tok), '--repeats', '5', '--num-tests', '30', '--out', out]
        print('>>', vname, sms, tok, flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith('rank'): print('  ', line.strip(), flush=True)
        if r.returncode != 0: print('FAILED', r.stderr[-3000:], flush=True)
subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')], check=True)
