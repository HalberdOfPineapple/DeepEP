"""E4: control-plane stores behind data. Runs the configurations E4 needs beyond what E0/E2 already recorded:
  - notify with a trivial data plane (64 tokens) at 16 SMs
  - full data plane (4096 tokens) at 16 and 32 SMs, baseline receive-array address
  - same with the receive array relocated by 4 MiB and by 4 MiB + 64 B into the unused hybrid signal region
The analysis also reads E0 (tokens 256/4096) and E2 (ratio 4) results, and the clock offsets from E1's ping-pong.
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
env = dict(os.environ, EP_DISABLE_GIN='1', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7'), PYTHONPATH=os.path.join(ROOT, 'common', 'pkg'))
plan = [('base', [], [(16, 64), (16, 256), (16, 4096), (32, 4096)]),
        ('incast4', ['--ratio-run'], [(16, 4096)]),
        ('reloc4M', ['--relocate-recv', str(4 << 20)], [(16, 4096), (32, 4096)]),
        ('reloc4M64', ['--relocate-recv', str((4 << 20) + 64)], [(16, 4096), (32, 4096)])]
for vname, vflags, configs in plan:
    extra = ['--ratio', '4', '--precise', '1'] if vflags == ['--ratio-run'] else []
    subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')] + ([] if vflags == ['--ratio-run'] else vflags), check=True)
    for sms, tok in configs:
        out = os.path.join(HERE, 'results', f'{vname}_sms{sms}_tok{tok}.pt')
        if os.path.exists(out): print('skip', out); continue
        cmd = ['taskset', '-c', '52-103', sys.executable, os.path.join(ROOT, 'common', 'run_dispatch.py'), '--num-sms', str(sms), '--num-tokens', str(tok), '--repeats', '8', '--num-tests', '20', '--out', out] + extra
        print('>>', vname, sms, tok, flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith('rank'): print('  ', line.strip(), flush=True)
        if r.returncode != 0: print('FAILED', r.stderr[-3000:], flush=True)
subprocess.run([sys.executable, os.path.join(ROOT, 'common', 'apply_trace_patch.py')], check=True)
