// E1: what does cp.async.bulk completion mean when the destination is a peer GPU?
// Standalone kernels, no NCCL, peers reached through CUDA P2P (UVA) pointers.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

#define CU_CHECK(x) TORCH_CHECK((x) == cudaSuccess, "CUDA error: ", cudaGetErrorString(cudaGetLastError()))

__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t cyc() { uint64_t t; asm volatile("mov.u64 %0, %clock64;" : "=l"(t)); return t; }
__device__ __forceinline__ int lane() { int l; asm("mov.u32 %0, %laneid;" : "=r"(l)); return l; }

__device__ __forceinline__ void bulk_store(void* dst, const void* smem_src, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :: "l"(dst), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_src))), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void st_rel_sys(int* p, int v) { asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rlx_sys(int* p, int v) { asm volatile("st.relaxed.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int ld_acq_sys(const int* p) { int v; asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

// mode: 0 wait_group 0 | 1 wait_group.read 0 | 2 wait_group 0 + fence.acq_rel.sys | 3 wait_group.read 0 + fence.acq_rel.sys
//       4 plain 16 B st.global per lane (bytes/512 per lane loop) + fence.acq_rel.sys | 5 wait_group 0 + fence.acq_rel.gpu
// One warp per slot; each warp has its own smem slot and its own destination region.
// out_cyc[(block*warps + warp)*iters + i] = cycles for (issue .. wait returned); out_ns same with globaltimer.
__global__ void __launch_bounds__(1024) bulk_latency_kernel(
    int8_t* dst_base, int64_t region_bytes, int bytes, int iters, int mode,
    long long* out_cyc, long long* out_ns, long long* calib /* [block][warp]: cyc0, ns0, cyc1, ns1 */) {
    extern __shared__ __align__(128) int8_t smem[];
    const int warp = threadIdx.x / 32, nwarps = blockDim.x / 32, l = lane();
    const int slot = blockIdx.x * nwarps + warp;
    int8_t* src = smem + static_cast<size_t>(warp) * ((bytes + 127) / 128 * 128);
    // fill smem once (generic proxy) then make it visible to the async proxy
    for (int i = l * 16; i < bytes; i += 32 * 16) if (i + 16 <= bytes) *reinterpret_cast<int4*>(src + i) = make_int4(slot, i, 0, 0);
    fence_proxy_async();
    __syncwarp();
    int8_t* region = dst_base + static_cast<int64_t>(slot) * region_bytes;
    const int64_t steps = region_bytes / bytes;
    const uint64_t c0 = cyc(), n0 = gtimer();
    for (int i = 0; i < iters; ++i) {
        int8_t* dst = region + (static_cast<int64_t>(i) % steps) * bytes;
        __syncwarp();
        const uint64_t t0 = cyc(), g0 = gtimer();
        if (mode == 4) {
            for (int off = l * 16; off < bytes; off += 32 * 16)
                if (off + 16 <= bytes) { const int4 v = *reinterpret_cast<int4*>(src + off); asm volatile("st.relaxed.sys.global.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(dst + off), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
            __syncwarp();
            if (l == 0) fence_sys();
        } else {
            if (l == 0) {
                bulk_store(dst, src, bytes);
                bulk_commit();
                if (mode == 1 || mode == 3) bulk_wait_read(); else bulk_wait_all();
                if (mode == 2 || mode == 3) fence_sys();
                if (mode == 5) fence_gpu();
            }
        }
        __syncwarp();
        const uint64_t t1 = cyc(), g1 = gtimer();
        if (l == 0) { out_cyc[static_cast<int64_t>(slot) * iters + i] = t1 - t0; out_ns[static_cast<int64_t>(slot) * iters + i] = g1 - g0; }
    }
    if (l == 0) { bulk_wait_all(); const uint64_t c1 = cyc(), n1 = gtimer(); calib[slot * 4 + 0] = c0; calib[slot * 4 + 1] = n0; calib[slot * 4 + 2] = c1; calib[slot * 4 + 3] = n1; }
}

// Issue n bulk stores back to back without waiting; record the cycle after each issue, then wait and record completion.
__global__ void __launch_bounds__(32) issue_rate_kernel(int8_t* dst_base, int64_t region_bytes, int bytes, int n,
                                                       long long* issue_cyc, long long* done_cyc) {
    extern __shared__ __align__(128) int8_t smem[];
    const int l = lane();
    for (int i = l * 16; i < bytes; i += 32 * 16) if (i + 16 <= bytes) *reinterpret_cast<int4*>(smem + i) = make_int4(i, 0, 0, 0);
    fence_proxy_async();
    __syncwarp();
    const int64_t steps = region_bytes / bytes;
    if (l == 0) {
        const uint64_t t0 = cyc();
        for (int i = 0; i < n; ++i) {
            bulk_store(dst_base + (static_cast<int64_t>(i) % steps) * bytes, smem, bytes);
            bulk_commit();
            issue_cyc[i] = cyc() - t0;
        }
        bulk_wait_all();
        done_cyc[0] = cyc() - t0;
    }
}

// Ping-pong between two GPUs with 4-byte flags. Each side runs this kernel with roles swapped.
// my_flag: local memory this side polls; peer_flag: pointer into the peer's memory this side writes.
// rec[i*4 + 0..3] = globaltimer at: before send, after seeing the reply (initiator) ; or at seeing ping, after sending pong (responder)
__global__ void pingpong_kernel(int* my_flag, int* peer_flag, int iters, int initiator, long long* rec_ns, long long* rec_cyc) {
    for (int i = 1; i <= iters; ++i) {
        if (initiator) {
            const uint64_t g0 = gtimer(), c0 = cyc();
            st_rlx_sys(peer_flag, i);
            while (ld_acq_sys(my_flag) < i) {}
            const uint64_t g1 = gtimer(), c1 = cyc();
            rec_ns[i * 2 + 0] = g0; rec_ns[i * 2 + 1] = g1; rec_cyc[i * 2 + 0] = c0; rec_cyc[i * 2 + 1] = c1;
        } else {
            while (ld_acq_sys(my_flag) < i) {}
            const uint64_t g0 = gtimer(), c0 = cyc();
            st_rlx_sys(peer_flag, i);
            const uint64_t g1 = gtimer(), c1 = cyc();
            rec_ns[i * 2 + 0] = g0; rec_ns[i * 2 + 1] = g1; rec_cyc[i * 2 + 0] = c0; rec_cyc[i * 2 + 1] = c1;
        }
    }
}

// Visibility probe: initiator does a bulk store of a pattern then (mode) wait variant, then st.relaxed.sys flag;
// responder spins on the flag, then reads the payload and records how many 16 B chunks already hold the pattern.
__global__ void visibility_sender(int8_t* peer_payload, int* peer_flag, int bytes, int iters, int mode) {
    extern __shared__ __align__(128) int8_t smem[];
    const int l = lane();
    for (int i = 1; i <= iters; ++i) {
        for (int off = l * 16; off < bytes; off += 32 * 16) if (off + 16 <= bytes) *reinterpret_cast<int4*>(smem + off) = make_int4(i, i, i, i);
        fence_proxy_async();
        __syncwarp();
        if (l == 0) {
            bulk_store(peer_payload, smem, bytes);
            bulk_commit();
            if (mode == 1) bulk_wait_read(); else bulk_wait_all();
            if (mode == 2) fence_sys();
            st_rlx_sys(peer_flag, i);
        }
        __syncwarp();
        // wait for the responder's ack (flag+1 in our own memory is handled by caller via peer_flag+1 convention)
        if (l == 0) { while (ld_acq_sys(peer_flag + 1) < i) {} }
        __syncwarp();
    }
}
__global__ void visibility_receiver(int8_t* payload, int* flag, int* peer_ack, int bytes, int iters, int* out_stale) {
    const int l = lane();
    for (int i = 1; i <= iters; ++i) {
        if (l == 0) { while (ld_acq_sys(flag) < i) {} }
        __syncwarp();
        int stale = 0;
        for (int off = l * 16; off < bytes; off += 32 * 16) {
            if (off + 16 <= bytes) { int vx; asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(vx) : "l"(payload + off) : "memory"); stale += (vx != i); }
        }
        for (int o = 16; o > 0; o >>= 1) stale += __shfl_xor_sync(0xffffffff, stale, o);
        if (l == 0) { out_stale[i] = stale; st_rlx_sys(peer_ack, i); }
        __syncwarp();
    }
}

// ---- host wrappers ----
static void set_dev(int d) { CU_CHECK(cudaSetDevice(d)); }

void enable_peer(int dev, int peer) {
    set_dev(dev);
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); return; }
    TORCH_CHECK(e == cudaSuccess, "enable peer failed: ", cudaGetErrorString(e));
}

std::vector<torch::Tensor> bulk_latency(int dev, torch::Tensor dst, int bytes, int iters, int mode, int blocks, int warps) {
    set_dev(dev);
    const int64_t region = dst.numel() / (static_cast<int64_t>(blocks) * warps) / bytes * bytes;
    TORCH_CHECK(region >= bytes, "destination too small");
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, dev);
    auto out_cyc = torch::zeros({static_cast<int64_t>(blocks) * warps * iters}, opts);
    auto out_ns = torch::zeros({static_cast<int64_t>(blocks) * warps * iters}, opts);
    auto calib = torch::zeros({static_cast<int64_t>(blocks) * warps * 4}, opts);
    const int smem = warps * ((bytes + 127) / 128 * 128);
    CU_CHECK(cudaFuncSetAttribute(bulk_latency_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    bulk_latency_kernel<<<blocks, warps * 32, smem, at::cuda::getCurrentCUDAStream(dev)>>>(
        dst.data_ptr<int8_t>(), region, bytes, iters, mode,
        reinterpret_cast<long long*>(out_cyc.data_ptr<int64_t>()), reinterpret_cast<long long*>(out_ns.data_ptr<int64_t>()), reinterpret_cast<long long*>(calib.data_ptr<int64_t>()));
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(dev)));
    return {out_cyc, out_ns, calib};
}

std::vector<torch::Tensor> issue_rate(int dev, torch::Tensor dst, int bytes, int n) {
    set_dev(dev);
    const int64_t region = dst.numel() / bytes * bytes;
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, dev);
    auto issue = torch::zeros({n}, opts); auto done = torch::zeros({1}, opts);
    const int smem = (bytes + 127) / 128 * 128;
    CU_CHECK(cudaFuncSetAttribute(issue_rate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    issue_rate_kernel<<<1, 32, smem, at::cuda::getCurrentCUDAStream(dev)>>>(dst.data_ptr<int8_t>(), region, bytes, n,
        reinterpret_cast<long long*>(issue.data_ptr<int64_t>()), reinterpret_cast<long long*>(done.data_ptr<int64_t>()));
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(dev)));
    return {issue, done};
}

// flags: int tensors on each device, >= 2 ints. Returns [recA_ns, recA_cyc, recB_ns, recB_cyc]
std::vector<torch::Tensor> pingpong(int devA, torch::Tensor flagA, int devB, torch::Tensor flagB, int iters) {
    auto optsA = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, devA);
    auto optsB = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, devB);
    auto recA_ns = torch::zeros({(iters + 1) * 2}, optsA), recA_cyc = torch::zeros({(iters + 1) * 2}, optsA);
    auto recB_ns = torch::zeros({(iters + 1) * 2}, optsB), recB_cyc = torch::zeros({(iters + 1) * 2}, optsB);
    flagA.zero_(); flagB.zero_();
    CU_CHECK(cudaDeviceSynchronize());
    set_dev(devB);
    pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream(devB)>>>(flagB.data_ptr<int>(), flagA.data_ptr<int>(), iters, 0,
        reinterpret_cast<long long*>(recB_ns.data_ptr<int64_t>()), reinterpret_cast<long long*>(recB_cyc.data_ptr<int64_t>()));
    CU_CHECK(cudaGetLastError());
    set_dev(devA);
    pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream(devA)>>>(flagA.data_ptr<int>(), flagB.data_ptr<int>(), iters, 1,
        reinterpret_cast<long long*>(recA_ns.data_ptr<int64_t>()), reinterpret_cast<long long*>(recA_cyc.data_ptr<int64_t>()));
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(devA)));
    set_dev(devB);
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(devB)));
    return {recA_ns, recA_cyc, recB_ns, recB_cyc};
}

// payload and flags live on devB; sender on devA. flagsB[0] = data flag, flagsB[1] = ack written by receiver into... we keep the ack on devB too
// (sender polls it remotely), which is enough for ordering the next round.
torch::Tensor visibility(int devA, int devB, torch::Tensor payloadB, torch::Tensor flagsB, int bytes, int iters, int mode) {
    auto out = torch::zeros({iters + 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devB));
    payloadB.zero_(); flagsB.zero_();
    CU_CHECK(cudaDeviceSynchronize());
    set_dev(devB);
    visibility_receiver<<<1, 32, 0, at::cuda::getCurrentCUDAStream(devB)>>>(payloadB.data_ptr<int8_t>(), flagsB.data_ptr<int>(),
        flagsB.data_ptr<int>() + 1, bytes, iters, out.data_ptr<int>());
    CU_CHECK(cudaGetLastError());
    set_dev(devA);
    const int smem = (bytes + 127) / 128 * 128;
    CU_CHECK(cudaFuncSetAttribute(visibility_sender, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    visibility_sender<<<1, 32, smem, at::cuda::getCurrentCUDAStream(devA)>>>(payloadB.data_ptr<int8_t>(), flagsB.data_ptr<int>(), bytes, iters, mode);
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(devA)));
    set_dev(devB);
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(devB)));
    return out;
}


// One side of a ping-pong with raw pointers (multi-process use: each process launches its own side on its own GPU).
std::vector<torch::Tensor> pingpong_side(int64_t my_flag_ptr, int64_t peer_flag_ptr, int iters, int initiator) {
    int dev = -1; CU_CHECK(cudaGetDevice(&dev));
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, dev);
    auto rec_ns = torch::zeros({(iters + 1) * 2}, opts), rec_cyc = torch::zeros({(iters + 1) * 2}, opts);
    pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream(dev)>>>(reinterpret_cast<int*>(my_flag_ptr), reinterpret_cast<int*>(peer_flag_ptr), iters, initiator,
        reinterpret_cast<long long*>(rec_ns.data_ptr<int64_t>()), reinterpret_cast<long long*>(rec_cyc.data_ptr<int64_t>()));
    CU_CHECK(cudaGetLastError());
    CU_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(dev)));
    return {rec_ns, rec_cyc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pingpong_side", &pingpong_side);
    m.def("enable_peer", &enable_peer);
    m.def("bulk_latency", &bulk_latency);
    m.def("issue_rate", &issue_rate);
    m.def("pingpong", &pingpong);
    m.def("visibility", &visibility);
}
