import os, sys, time, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def worker(lr, n):
    from deep_ep.utils.envs import init_dist
    rank, world, group = init_dist(lr, n, seed=0)
    from clocksync import ClockSync
    cs = ClockSync(group, rank, world)
    m0 = cs.measure(); t0 = time.time()
    time.sleep(20)
    m1 = cs.measure(); t1 = time.time()
    if rank == 0:
        for k in sorted(m0['offset']):
            if k[0] < k[1]:
                print(f'pair {k}: offset {m0["offset"][k]/1e6:12.6f} ms, rtt {m0["rtt"][k]:.0f} ns; after {t1-t0:.1f} s offset changed by {(m1["offset"][k]-m0["offset"][k]):8.0f} ns -> drift {(m1["offset"][k]-m0["offset"][k])/(t1-t0):8.1f} ns/s')
        # triangle closure
        o = m0['offset']; errs = [abs(o[(a,b)] + o[(b,c)] - o[(a,c)]) for a in range(world) for b in range(world) for c in range(world) if len({a,b,c})==3]
        print('triangle closure max/median ns:', max(errs), sorted(errs)[len(errs)//2])
    dist.destroy_process_group()
if __name__ == '__main__':
    mp.spawn(worker, args=(4,), nprocs=4)
