"""Blind case generator ("the attacker").

Runs the standard 4-rank dispatch workload on GPUs 4-7 with ONE hidden perturbation, then writes:
  cases/<name>/run.pt        scrubbed trace (no perturbation flags, no background/overlap records)
  cases/<name>/nvlink.json   NVLink TX/RX byte totals per GPU before and after the whole run (one read each, no perturbation)
  sealed/<name>.json         the ground truth (the diagnoser must not read this directory)
Perturbations that are external processes (foreign traffic, compute neighbour) are started before and stopped after the run.
"""
import os, sys, json, time, subprocess, random, shutil, argparse
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
ENV = dict(os.environ, EP_DISABLE_GIN='1', CUDA_VISIBLE_DEVICES='4,5,6,7', PYTHONPATH=os.path.join(ROOT, 'common', 'pkg'))
BASE = ['--num-sms', '16', '--num-tokens', '4096', '--repeats', '5', '--host-ts', '1', '--num-tests', '20']

ATTACKS = {
    'none':        dict(kind='control', desc='no perturbation', flags=[], external=None),
    'nccl':        dict(kind='foreign_traffic', desc='separate 4-process NCCL all_to_all (64 MB) job on GPUs 4-7 during the whole run', flags=[], external=('nccl', 64)),
    'copy_gpu5':   dict(kind='compute_neighbour', desc='separate process on GPU 5 (rank 1) running 32 MB HBM copies during the whole run', flags=[], external=('copy', 5)),
    'gemm_gpu6':   dict(kind='compute_neighbour', desc='separate process on GPU 6 (rank 2) running 2048^3 GEMMs during the whole run', flags=[], external=('gemm', 6)),
    'hostskew_r2': dict(kind='host_skew', desc='rank 2 host busy-waits 300 us before each traced dispatch', flags=['--skew-rank', '2', '--skew-us', '300', '--skew-mode', 'host'], external=None),
    'gpuskew_r3':  dict(kind='gpu_skew', desc='rank 3 runs a 400 us spin kernel on its stream before each traced dispatch', flags=['--skew-rank', '3', '--skew-us', '400', '--skew-mode', 'gpu'], external=None),
    'incast3':     dict(kind='routing_incast', desc='routing ratio 3: rank 0 receives 3x the tokens of other ranks', flags=['--ratio', '3'], external=None),
    'fewsms':      dict(kind='misconfiguration', desc='num_sms 4 instead of 16', flags=[], external=None, override={'--num-sms': '4'}),
}

def nvlink_totals():
    out = subprocess.run(['nvidia-smi', 'nvlink', '-gt', 'd'], capture_output=True, text=True).stdout
    tot = {}; gpu = None
    for line in out.splitlines():
        if line.startswith('GPU'): gpu = int(line.split(':')[0].split()[1]); tot[gpu] = {'tx': 0, 'rx': 0}
        elif 'Data Tx' in line: tot[gpu]['tx'] += int(line.split(':')[-1].split()[0])
        elif 'Data Rx' in line: tot[gpu]['rx'] += int(line.split(':')[-1].split()[0])
    return {g: v for g, v in tot.items() if g in (4, 5, 6, 7)}

def scrub(src, dst):
    d = torch.load(src, weights_only=False)
    for k in ('bg', 'bg_ranks', 'bg_iters', 'skew_rank', 'skew_us', 'skew_mode', 'overlap', 'nvml', 'ratio', 'precise', 'seed', 'out'):
        d['args'].pop(k, None)
    for g in d['ranks']:
        for k in ('bg', 'overlap', 'nvml'): g.pop(k, None)
    torch.save(d, dst)

def run_case(name, attack, outdir, repeat_tag=''):
    os.makedirs(outdir, exist_ok=True)
    stop = os.path.join(HERE, f'.stop_{name}'); 
    if os.path.exists(stop): os.remove(stop)
    procs = []
    if attack['external']:
        kind, arg = attack['external']
        if kind == 'nccl':
            procs.append(subprocess.Popen([PY, os.path.join(HERE, 'noise', 'noise_nccl.py'), stop, str(arg)], env=dict(os.environ, CUDA_VISIBLE_DEVICES='4,5,6,7'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        else:
            procs.append(subprocess.Popen([PY, os.path.join(HERE, 'noise', 'noise_compute.py'), stop, kind], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(arg)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(8)  # let the noise reach steady state
    flags = list(BASE)
    for k, v in (attack.get('override') or {}).items():
        i = flags.index(k); flags[i + 1] = v
    raw = os.path.join(HERE, 'sealed', f'{name}_raw.pt')
    before = nvlink_totals(); t0 = time.time()
    cmd = ['taskset', '-c', '52-103', PY, os.path.join(ROOT, 'common', 'run_dispatch.py')] + flags + attack['flags'] + ['--out', raw]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    t1 = time.time(); after = nvlink_totals()
    open(stop, 'w').write('stop')
    for p in procs: p.wait(timeout=120)
    if os.path.exists(stop): os.remove(stop)
    if r.returncode != 0:
        print('FAILED', name, r.stderr[-2000:]); return False
    scrub(raw, os.path.join(outdir, 'run.pt'))
    json.dump(dict(before=before, after=after, wall_s=t1 - t0, unit='KiB, summed over the 18 links of each GPU'), open(os.path.join(outdir, 'nvlink.json'), 'w'), indent=1)
    truth = dict(case=name, kind=attack['kind'], desc=attack['desc'], flags=attack['flags'], external=attack['external'], override=attack.get('override'), wall_s=t1 - t0, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(truth, open(os.path.join(HERE, 'sealed', f'{name}.json'), 'w'), indent=1)
    print('done', name, f'{t1 - t0:.0f}s', [l for l in r.stdout.splitlines() if l.startswith('rank 0')][:1])
    return True

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--baseline', action='store_true'); ap.add_argument('--cases', type=str, default=''); ap.add_argument('--shuffle-seed', type=int, default=7)
    a = ap.parse_args()
    subprocess.run([PY, os.path.join(ROOT, 'common', 'apply_trace_patch.py')], check=True, stdout=subprocess.DEVNULL)
    if a.baseline:
        run_case('baseline', ATTACKS['none'], os.path.join(HERE, 'baseline'))
    names = a.cases.split(',') if a.cases else list(ATTACKS)
    rng = random.Random(a.shuffle_seed); order = list(names); rng.shuffle(order)
    mapping = {}
    for i, n in enumerate(order):
        cid = f'case{i + 1:02d}'; mapping[cid] = n
        run_case(cid, ATTACKS[n], os.path.join(HERE, 'cases', cid))
    json.dump(mapping, open(os.path.join(HERE, 'sealed', 'mapping.json'), 'w'), indent=1)
    print('mapping sealed')
