"""Foreign NVLink traffic: 4 processes on the visible GPUs run NCCL all_to_all_single in a loop until a stop file appears."""
import os, sys, time, torch, torch.distributed as dist, torch.multiprocessing as mp
def worker(rank, world, stop_file, mb):
    os.environ['MASTER_ADDR'] = '127.0.0.1'; os.environ['MASTER_PORT'] = '8477'
    torch.cuda.set_device(rank); dist.init_process_group('nccl', rank=rank, world_size=world)
    x = torch.randn(mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device='cuda'); y = torch.empty_like(x)
    while not os.path.exists(stop_file):
        for _ in range(8): dist.all_to_all_single(y, x)
        torch.cuda.synchronize()
    dist.destroy_process_group()
if __name__ == '__main__':
    stop_file, mb = sys.argv[1], int(sys.argv[2]); world = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(','))
    mp.spawn(worker, args=(world, stop_file, mb), nprocs=world, join=True)
