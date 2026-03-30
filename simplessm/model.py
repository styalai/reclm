import torch 
import torch.nn as nn

@dataclass
class Config:
    seq_len: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_embd: int = 768
    dropout: float = 0.0 

class SSM(nn.Module):
    """
    Simple block for SSM ht+1 = Aht + Bxt
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.A = nn.Parameter(torch.emty(config.n_embd, config.n_embd))
        self.B = nn.Parameter(torch.emty(config.n_embd, config.n_embd))
        self.h = self.register_buffer("h")
