"""Compute neighbour on one GPU: short HBM copies or GEMMs in a loop until a stop file appears."""
import os, sys, time, torch
stop_file, kind = sys.argv[1], sys.argv[2]
torch.cuda.set_device(0)
if kind == 'copy':
    S = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device='cuda'); D = torch.empty_like(S)
    def k(): D.copy_(S)
else:
    A = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda'); B = torch.randn_like(A); C = torch.empty_like(A)
    def k(): torch.matmul(A, B, out=C)
while not os.path.exists(stop_file):
    for _ in range(200): k()
    torch.cuda.synchronize()
