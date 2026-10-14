"""E5: compute interference. A stream of short HBM-bound copies or compute-bound GEMMs runs beside the traced dispatch, on all ranks or on rank 0 only."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from sweep import run_cases
base = ['--num-sms', 16, '--num-tokens', 4096, '--repeats', 3, '--flush-l2', 0, '--num-tests', 20]
cases = [
    ('none', (), base + ['--bg', 'none']),
    ('copy_all', (), base + ['--bg', 'copy', '--bg-ranks', 'all', '--bg-iters', 120]),
    ('copy_rank0', (), base + ['--bg', 'copy', '--bg-ranks', '0', '--bg-iters', 120]),
    ('gemm_all', (), base + ['--bg', 'gemm', '--bg-ranks', 'all', '--bg-iters', 120]),
    ('gemm_rank0', (), base + ['--bg', 'gemm', '--bg-ranks', '0', '--bg-iters', 120]),
    ('copy_all_tok256', (), ['--num-sms', 16, '--num-tokens', 256, '--repeats', 5, '--flush-l2', 0, '--bg', 'copy', '--bg-ranks', 'all', '--bg-iters', 30]),
    ('none_tok256', (), ['--num-sms', 16, '--num-tokens', 256, '--repeats', 5, '--flush-l2', 0, '--bg', 'none']),
]
run_cases(HERE, cases)
