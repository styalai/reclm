import tilelang
import tilelang.language as T
import torch, math

@tilelang.jit
def upsweep_kernel(SEQ, H, BLOCK_SEQ, BLOCK_H, dtype=T.bfloat16, threads=256):
    """
    great article: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    l = 1 # 1, 2, 4, 8 -> 0, 1, 3, 7
    for i in range(log2(seq)):
        l = i ** 2 -1 # index de beginning of the scan 

        # step between 2 beginning of pair: 2, 4, 8 -> (l+1)*2
        # step between 2 of the same pairs: l + 1
        # get every beginning of pair
        for j in range(l, seq-1, (l+1)*2):
            a0 = A[l]
            u0 = U[l]
            a1 = A[l+1]
            u1 = A[l+1]

            A[l+1] = a0*a1
            U[l+1] = a1*u0 + u1

    from article:
    1: for d = 0 to log2 n – 1 do
    2: for all k = 0 to n – 1 by 2**d+1 in parallel do
    3: x[k + 2 d+1 – 1] = x[k + 2 d – 1] + x[k + 2 d +1 – 1]
            
    """

    @T.prim_func # if A is always the same for each state, need to duplicate it 
    def upsweep(A: T.Tensor((SEQ, H), dtype), U: T.Tensor((SEQ, H), dtype)): # type: ignore
        with T.Kernel(T.ceildiv(H, BLOCK_H), threads=threads) as bh:
            for i in T.Serial(math.log2(SEQ)): # parcour en depth -> index de debut du scan
                l = i*i -1

                for j in T.Parallel(len(range(l, SEQ-1, (l+1)*2))):
                    a0 = A[l, bh]
                    u0 = U[l, bh]
                    a1 = A[l+1, bh]
                    u1 = U[l+1, bh]

                    A[l+1, bh] = a0*a1
                    U[l+1, bh] = a1*u0 + u1

    return upsweep


if __name__ == "__main__":
    k = upsweep_kernel(4, 8, 1, 1)