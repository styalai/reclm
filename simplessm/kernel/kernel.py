import tilelang
import tilelang.language as T
import torch

def upsweep(SEQ, H, BLOCK_SEQ, BLOCK_H, dtype=T.bfloat16, threads=256):
    """
    great article: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    l = 1 # 1, 2, 4, 8 -> 0, 1, 3, 7
    for i in range(log2(seq)):
        l = i ** 2 -1 # index de beginning of the scan 

        # step between 2 beginning of pair: 2, 4, 8
        # step between 2 of the same pairs: l + 1
        # get evry beginning of pair
        for 
            
    """

    @T.prim_func
    def upsweep(A: T.Tensor((H,), dtype), U: T.Tensor((SEQ, H), dtype), dtype): # type: ignore
        with T.Kernel(T.ceildiv(SEQ, BLOCK_SEQ), T.ceildiv(H, BLOCK_H), threads=threads) as (bs, bh):
            
