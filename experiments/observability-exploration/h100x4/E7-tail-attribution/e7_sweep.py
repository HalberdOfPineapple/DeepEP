"""E7: tail attribution. Uniform and skewed routing with several seeds; the analysis also reads E2's ratio sweep."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from sweep import run_cases
cases = []
for ratio in (1, 3):
    for seed in (1, 2):
        cases.append((f'ratio{ratio}_seed{seed}', (), ['--num-sms', 16, '--num-tokens', 4096, '--repeats', 3, '--no-kineto', 1, '--ratio', ratio, '--seed', seed]))
run_cases(HERE, cases)
