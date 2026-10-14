"""E9: decode critical path. 128 tokens per rank, many repeats, host timestamps mapped onto the GPU clock; CPU sync on/off, padded counters, 32 SMs."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from sweep import run_cases
base = ['--num-tokens', 128, '--repeats', 100, '--host-ts', 1, '--num-tests', 50]
cases = [
    ('base_sms16', (), base + ['--num-sms', 16, '--cpu-sync', 1]),
    ('nocpusync_sms16', (), base + ['--num-sms', 16, '--cpu-sync', 0]),
    ('base_sms32', (), base + ['--num-sms', 32, '--cpu-sync', 1]),
    ('base_sms4', (), base + ['--num-sms', 4, '--cpu-sync', 1]),
    ('padded_sms16', ('--pad-counters',), base + ['--num-sms', 16, '--cpu-sync', 1]),
]
run_cases(HERE, cases)
