"""Shared loader for run_dispatch.py outputs: turns raw int64 trace tensors into per-rank, per-repeat dictionaries.

Each repeat record (one per rank) has:
  sm[e][s]     SM-level event e (trace_layout.SM_EVENTS) on SM s, ns on this GPU's clock
  warp[w]      (first_entry, last_done, tokens, sm) per dispatch warp
  tok          per-token record tensor [T, 64] filtered to tokens this rank handled (T_ISSUE > 0)
  off(r2)      clock offset: value_on_rank_r2_clock - value_on_this_rank_clock  (mean of the bracketing calibrations)
"""
import torch, numpy as np
import trace_layout as TL

def load(path):
    d = torch.load(path, weights_only=False); a = d['args']; NR = d['ranks'][0]['num_ranks']; T = a['num_tokens']; sms = a['num_sms']
    out = dict(args=a, NR=NR, T=T, sms=sms, ranks={}, nrep=len(d['ranks'][0]['traces']), raw=d)
    for g in d['ranks']:
        r = g['rank']; reps = []
        for i, tr in enumerate(g['traces']):
            sm = {e: [int(tr[TL.SM + s * 32 + e]) for s in range(sms)] for e in TL.SM_EVENTS}
            warp = tr[TL.WARP:TL.WARP + 4 * sms * 15].view(-1, 4).clone()
            rec = tr[TL.TOK:TL.TOK + TL.TOK_STRIDE * T].view(T, TL.TOK_STRIDE); rec = rec[rec[:, TL.T_ISSUE] > 0].clone()
            clocks = g.get('clocks'); off = {}
            if clocks:
                cb, ca = clocks[i]['before']['offset'], clocks[i]['after']['offset']
                for (x, y), v in cb.items():
                    if x == r: off[y] = (v + ca[(x, y)]) / 2
                off[r] = 0.0
            reps.append(dict(sm=sm, warp=warp, tok=rec, off=off, host=(g.get('host_ts') or [None] * 99)[i], bg=g.get('bg'), overlap=g.get('overlap'),
                             nvml=(g.get('nvml') or [None] * 99)[i] if g.get('nvml') else None))
        out['ranks'][r] = dict(reps=reps, t_untraced_s=g['t_untraced_s'], t_traced_s=g['t_traced_s'], dst_counts=g['dst_counts'], k_mean=g['k_mean'], topk_idx=g['topk_idx'])
    return out

def to_rank0_clock(run, rep_idx, r, value_ns):
    """Express a stamp taken on rank r (repeat rep_idx) on rank 0's clock."""
    rep = run['ranks'][r]['reps'][rep_idx]
    # off[y] on rank r = clock_y - clock_r; we want clock_0: value + (clock_0 - clock_r)
    if r == 0: return value_ns
    rep0 = run['ranks'][0]['reps'][rep_idx]
    return value_ns - rep0['off'][r]  # clock_r - clock_0 = off_0[r]  -> value_on_0 = value - off_0[r]

def phase_durations(rep):
    """Durations (us) of the phases of one rank's dispatch from its SM-0 stamps and warp records."""
    sm = rep['sm']; s0 = {e: sm[e][0] for e in sm}
    wd = rep['warp'][:, 1][rep['warp'][:, 1] > 0]; wf = rep['warp'][:, 0][rep['warp'][:, 0] > 0]
    last_warp = int(wd.max()) if wd.numel() else s0[9]; first_warp = int(wd.min()) if wd.numel() else s0[9]
    return dict(entry=s0[0], open_arrive=s0[1], open_release=max(sm[2]), notify_c=s0[3], notify_e=s0[5], notify_f=s0[6], notify_h=s0[7],
                first_warp_done=first_warp, last_warp_done=last_warp, close_arrive=max(sm[9]), close_release=max(sm[10]), end=max(sm[11]),
                A_us=(max(sm[2]) - s0[0]) / 1e3, A_wait_us=(max(sm[2]) - s0[1]) / 1e3, B_us=(s0[6] - max(sm[2])) / 1e3 if s0[6] else float('nan'),
                data_us=(last_warp - max(sm[2])) / 1e3, tail_us=(last_warp - first_warp) / 1e3, E_wait_us=(max(sm[10]) - last_warp) / 1e3,
                close_us=(max(sm[10]) - max(sm[9])) / 1e3, F_us=(max(sm[11]) - max(sm[10])) / 1e3, span_us=(max(sm[11]) - s0[0]) / 1e3)

def lane_latencies(rep, K):
    rec = rep['tok']; li = rec[:, TL.T_LANE_ISSUE:TL.T_LANE_ISSUE + K]; ld = rec[:, TL.T_LANE_DONE:TL.T_LANE_DONE + K]; dst = rec[:, TL.T_LANE_DST:TL.T_LANE_DST + K]
    m = (li > 0) & (ld > 0)
    return (ld[m] - li[m]).double().numpy(), dst[m].numpy(), li[m].numpy(), ld[m].numpy()

def token_parts(rep):
    rec = rep['tok']
    return dict(slot_wait=(rec[:, TL.T_SLOTWAIT_END] - rec[:, TL.T_SLOTWAIT_START]).double().numpy(),
                prepare=(rec[:, TL.T_LOAD_DONE] - rec[:, TL.T_SLOTWAIT_END]).double().numpy(),
                issue=(rec[:, TL.T_ISSUE] - rec[:, TL.T_LOAD_DONE]).double().numpy(),
                iteration=(rec[:, TL.T_ISSUE] - rec[:, TL.T_SLOTWAIT_START]).double().numpy())

def atomic_latencies(rep, K):
    rec = rep['tok']; ae = rec[:, TL.T_LANE_ATOMIC_END:TL.T_LANE_ATOMIC_END + K]; am = ae > 0
    return (ae[am] - rec[:, TL.T_ATOMIC_START].unsqueeze(1).expand_as(ae)[am]).double().numpy()
