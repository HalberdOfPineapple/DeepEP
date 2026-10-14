"""Shared sweep runner: each case = (name, patch_flags, run_dispatch_args). Skips cases whose .pt exists."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run_cases(here, cases):
    os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    env = dict(os.environ, EP_DISABLE_GIN='1', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7'), PYTHONPATH=os.path.join(ROOT, 'common', 'pkg'))
    cur_patch = None
    for name, pflags, rargs in cases:
        out = os.path.join(here, 'results', f'{name}.pt')
        if os.path.exists(out): print('skip', out, flush=True); continue
        if pflags != cur_patch:
            subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')] + list(pflags), check=True); cur_patch = pflags
        cmd = ['taskset', '-c', '52-103', sys.executable, os.path.join(ROOT, 'common', 'run_dispatch.py')] + [str(v) for v in rargs] + ['--out', out]
        print('>>', name, ' '.join(str(v) for v in rargs), flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith('rank'): print('  ', line.strip(), flush=True)
        if r.returncode != 0: print('FAILED', name, r.stderr[-3000:], flush=True)
    if cur_patch != ():
        subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')], check=True)
