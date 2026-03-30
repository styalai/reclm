import torch
from model import SSM, Config

conf = Config(
    seq_len=1024,
    vocab_size=50304,
    n_layer=12,
    n_embd=768,
    dropout=0.0, 
)
model = SSM(conf).to("cuda").eval()
print(model(torch.randn(1, 768).cuda()))
print(model.h)

