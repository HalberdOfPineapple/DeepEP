"""E8: two kernels on the same links. The traced dispatch runs alone, beside a DeepEP combine on a second buffer, and beside a NCCL all-to-all; per-link NVLink counters are read before and after."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from sweep import run_cases
base = ['--num-sms', 16, '--num-tokens', 4096, '--repeats', 5, '--no-kineto', 1, '--nvml', 1]
cases = [
    ('alone', (), base + ['--overlap', 'none']),
    ('combine', (), base + ['--overlap', 'combine']),
    ('nccl6', (), base + ['--overlap', 'nccl', '--bg-iters', 6]),
    ('nccl16', (), base + ['--overlap', 'nccl', '--bg-iters', 16]),
]
run_cases(HERE, cases)
