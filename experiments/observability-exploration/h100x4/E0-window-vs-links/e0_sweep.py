"""E0: dispatch throughput against concurrency (num_sms) and token count, uniform routing, unmodified kernel logic.

Runs common/run_dispatch.py for every (num_sms, num_tokens) pair and stores one .pt per pair under results/.
"""
import itertools, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
sms_list = [2, 4, 8, 12, 16, 24, 32]
tok_list = [256, 1024, 4096]
env = dict(os.environ, EP_DISABLE_GIN='1', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7'), PYTHONPATH=os.path.join(ROOT, 'common', 'pkg'))
for sms, tok in itertools.product(sms_list, tok_list):
    out = os.path.join(HERE, 'results', f'sms{sms}_tok{tok}.pt')
    if os.path.exists(out):
        print('skip', out); continue
    cmd = ['taskset', '-c', '52-103', sys.executable, os.path.join(ROOT, 'common', 'run_dispatch.py'), '--num-sms', str(sms), '--num-tokens', str(tok),
           '--repeats', '3', '--num-tests', '20', '--out', out]
    print('>>', ' '.join(cmd[3:]), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith('rank') or line.startswith('auto'):
            print('  ', line, flush=True)
    if r.returncode != 0:
        print('FAILED', r.stderr[-3000:], flush=True)
