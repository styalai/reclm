import torch, math

def upsweep(x: torch.Tensor, seq): # seq must be even
    """
    from article:
    1: for d = 0 to log2 n – 1 do
    2: for all k = 0 to n – 1 by 2**d+1 in parallel do
    3: x[k + 2 d+1 – 1] = x[k + 2 d – 1] + x[k + 2 d]
    """
    for depth in range(int(math.log2(seq))): # 0, 1, 2, 3...
        # beginning of first pair: 0, 2, 4, 8...
        print("d: ", depth)
        for i in range(2**depth-1, seq-1, 2**(depth+1)): # for every beggining of pair
            print(i)

if __name__ == "__main__":
    x = torch.tensor([0, 1, 2, 3])
    o = upsweep(x, x.shape[0])


