import torch, math

def upsweep(x: torch.Tensor, seq): # seq must be even
    """
    from article: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    1: for d = 0 to log2 n – 1 do
    2: for all k = 0 to n – 1 by 2**d+1 in parallel do
    3: x[k + 2**d+1 – 1] = x[k + 2**d – 1] + x[k + 2**d]
    """
    for depth in range(int(math.log2(seq))): # 0, 1, 2, 3...
        #print("d: ", depth)
        # beginning of first pair: 0, 2, 4, 8...
        stride = 2**(depth+1)
        for i in range(2**depth-1, seq-1, stride): # for every beggining of pair in this depth
            #print(i, i+2**depth)
            x[i+2**depth] = x[i] + x[i+2**depth]

        if i+2**depth < seq-1:
            i = i+2**depth
            dist = (seq-1) - (i)
            x[i+dist] = x[i] + x[i+dist]
            #print("last: ", i, i+dist)


    return x

def downsweep(x: torch.Tensor, seq):
    x[seq-1] = 0
    for depth in range(int(math.log2(seq))-1, -1, -1):
        #print("d:", depth)
        stride = -2**depth -1
        print("s", stride)
        for i in range(seq, seq-2**depth, stride):
            i = i-1
            print(i)
    return x
        


if __name__ == "__main__":
    x = torch.tensor([0, 1, 2, 3, 4, 5]) # -> 15 [0, 1s, 2, 6s, 4, 15s]
    o = upsweep(x, x.shape[0])
    s = downsweep(o, x.shape[0])
    print(o)

