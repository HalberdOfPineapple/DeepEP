"""Host-side helpers for the dispatch trace region (see apply_trace_patch.py for the layout)."""
import torch

MAGIC = 0x4550545243
HDR, SM, SEND, RECV, WARP, TOK, TOK_STRIDE = 0, 16, 1040, 3088, 5136, 8192, 64
SM_EVENTS = {0: 'kernel_entry', 1: 'open_barrier_arrive', 2: 'open_barrier_release', 3: 'notify_C_done', 4: 'notify_D_done',
             5: 'notify_E_issued', 6: 'notify_F_done', 7: 'notify_H_done', 9: 'close_barrier_arrive', 10: 'close_barrier_release', 11: 'kernel_end'}
# per-token record (int64 index)
T_ID, T_SLOTWAIT_START, T_SLOTWAIT_END, T_ATOMIC_START, T_LOAD_DONE, T_ISSUE, T_DSTMASK = 0, 1, 2, 3, 5, 6, 7
T_LANE_ISSUE, T_LANE_DONE, T_LANE_DST, T_LANE_SLOT, T_LANE_ATOMIC_END = 8, 16, 24, 32, 40


def alloc_trace(num_local_experts: int, num_tokens: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (stats_view, trace_i64). stats_view is what is passed as cumulative_local_expert_recv_stats."""
    n_i64 = TOK + TOK_STRIDE * num_tokens
    assert num_local_experts % 2 == 0
    big = torch.zeros(num_local_experts + 2 * n_i64, dtype=torch.int32, device=device)
    stats = big[:num_local_experts]
    trace = big[num_local_experts:].view(torch.int64)
    trace[HDR] = MAGIC; trace[HDR + 1] = num_tokens
    return stats, trace


def reset_trace(trace: torch.Tensor, num_tokens: int):
    trace[SM:].zero_(); trace[HDR] = MAGIC; trace[HDR + 1] = num_tokens


def decode(trace_cpu: torch.Tensor, num_tokens: int, num_sms: int, num_topk: int, num_ranks: int) -> dict:
    """Turn one rank's trace into plain python structures (ns timestamps as ints)."""
    t = trace_cpu
    sm = {SM_EVENTS[e]: [int(t[SM + s * 32 + e]) for s in range(num_sms)] for e in SM_EVENTS}
    tok = t[TOK:TOK + TOK_STRIDE * num_tokens].view(num_tokens, TOK_STRIDE)
    lane_issue = tok[:, T_LANE_ISSUE:T_LANE_ISSUE + num_topk]
    lane_done = tok[:, T_LANE_DONE:T_LANE_DONE + num_topk]
    lane_dst = tok[:, T_LANE_DST:T_LANE_DST + num_topk]
    lane_atomic_end = tok[:, T_LANE_ATOMIC_END:T_LANE_ATOMIC_END + num_topk]
    valid = (lane_issue > 0) & (lane_done > 0)
    return dict(
        sm=sm,
        send=t[SEND:SEND + num_ranks * 0 + num_ranks + 256 * 0 + (t[HDR + 2].item() if False else 0)] if False else None,
        tok=tok, lane_issue=lane_issue, lane_done=lane_done, lane_dst=lane_dst, lane_atomic_end=lane_atomic_end, valid=valid,
        warp=t[WARP:WARP + 4 * 512].view(512, 4),
    )
