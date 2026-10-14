import os, sys, time, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def worker(lr, n):
    from deep_ep.utils.envs import init_dist
    rank, world, group = init_dist(lr, n, seed=0)
    from clocksync import ClockSync
    cs = ClockSync(group, rank, world)
    idx = 1
    cs.flags.zero_(); torch.cuda.synchronize(); dist.barrier(group)
    if rank == 1:
        rec, _ = cs.ext.pingpong_side(cs.flags.data_ptr() + 4*idx, cs.peer_ptrs[0] + 4*idx, 400, 0); r = rec.cpu().view(-1,2)
    elif rank == 0:
        rec, _ = cs.ext.pingpong_side(cs.flags.data_ptr() + 4*idx, cs.peer_ptrs[1] + 4*idx, 400, 1); r = rec.cpu().view(-1,2)
    else: r = None
    dist.barrier(group)
    g = [None]*world; dist.all_gather_object(g, r, group=group)
    if rank == 0:
        ra, rb = g[0], g[1]
        print('ra dtype', ra.dtype, 'rows', ra[50:53].tolist()); print('rb rows', rb[50:53].tolist())
        d = ((rb[50:, 0] - ra[50:, 0]) - (ra[50:, 1] - rb[50:, 1]))
        print('d dtype', d.dtype, d[:5].tolist(), 'd/2 dtype', (d/2).dtype, (d//2)[:5].tolist())
        print('rtt', (ra[50:,1]-ra[50:,0])[:5].tolist())
    dist.destroy_process_group()
if __name__ == '__main__':
    mp.spawn(worker, args=(4,), nprocs=4)
