"""E2: destination incast. Sweep the unbalanced ratio with the baseline kernel and with the wait_group.read variant.

Each run: 4 ranks, 4096 tokens, 16 SMs (DeepEP's own choice is 20; 16 keeps the per-SM queue below the link so that
destination effects are not masked by sender-side saturation), precise unbalanced generator (rank 0 is the hot rank).
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
ratios = [1.0, 1.5, 2.0, 3.0, 4.0]
variants = [('base', []), ('waitread', ['--wait-read'])]
env = dict(os.environ, EP_DISABLE_GIN='1', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7'), PYTHONPATH=os.path.join(ROOT, 'common', 'pkg'))
sms = int(os.environ.get('E2_SMS', '16'))
for vname, vflags in variants:
    subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')] + vflags, check=True)
    for ratio in ratios:
        out = os.path.join(HERE, 'results', f'{vname}_ratio{ratio}_sms{sms}.pt')
        if os.path.exists(out):
            print('skip', out); continue
        cmd = ['taskset', '-c', '52-103', sys.executable, os.path.join(ROOT, 'common', 'run_dispatch.py'), '--num-sms', str(sms), '--num-tokens', '4096',
               '--ratio', str(ratio), '--precise', '1', '--repeats', '3', '--num-tests', '20', '--out', out]
        print('>>', vname, 'ratio', ratio, flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith('rank') or line.startswith('auto'):
                print('  ', line.strip(), flush=True)
        if r.returncode != 0:
            print('FAILED', r.stderr[-3000:], flush=True)
subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')], check=True)  # back to baseline
