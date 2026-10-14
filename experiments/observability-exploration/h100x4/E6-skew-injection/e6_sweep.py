"""E6: skew injection. Rank 0 is delayed before each traced dispatch by a GPU spin kernel (or a host sleep); the cross-rank timeline shows where the others wait."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from sweep import run_cases
base = ['--num-sms', 16, '--num-tokens', 4096, '--repeats', 5, '--no-kineto', 1, '--host-ts', 1]
cases = [('gpu_0', (), base + ['--skew-rank', 0, '--skew-us', 0, '--skew-mode', 'gpu'])]
for us in (20, 100, 500, 2000):
    cases.append((f'gpu_{us}', (), base + ['--skew-rank', 0, '--skew-us', us, '--skew-mode', 'gpu']))
cases.append(('host_100', (), base + ['--skew-rank', 0, '--skew-us', 100, '--skew-mode', 'host']))
cases.append(('gpu_100_tok256', (), ['--num-sms', 16, '--num-tokens', 256, '--repeats', 5, '--no-kineto', 1, '--host-ts', 1, '--skew-rank', 0, '--skew-us', 100, '--skew-mode', 'gpu']))
cases.append(('gpu_0_tok256', (), ['--num-sms', 16, '--num-tokens', 256, '--repeats', 5, '--no-kineto', 1, '--host-ts', 1, '--skew-rank', 0, '--skew-us', 0, '--skew-mode', 'gpu']))
run_cases(HERE, cases)
