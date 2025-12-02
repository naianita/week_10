import hashlib

import torch


EMBED_DIM = 384


def _hashed_embedding(text: str, dim: int = EMBED_DIM, device: torch.device | None = None) -> torch.Tensor:
    """
    Lightweight, deterministic text embedding that avoids loading large
    transformer models. We hash the input text to seed a small random
    vector in R^dim.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(dim, generator=gen, device=device)


def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """
    Return a fixed-size embedding for an operator description or query.

    This replaces the original SentenceTransformer-based embedding with a
    lightweight hashed embedding to avoid large model downloads and high
    memory usage on constrained machines.
    """
    return _hashed_embedding(sentence)


class SentenceEncoder(torch.nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.dim = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, sentence: str) -> torch.Tensor:
        return _hashed_embedding(sentence, dim=self.dim, device=self.device)

def sample_operators(probs: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    device = probs.device
    probs = probs.detach()
    
    num_ops = probs.size(0)
    if num_ops == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    selected = torch.tensor([], dtype=torch.long, device=device)
    cumulative = 0.0
    remaining = torch.arange(num_ops, device=device)
    
    while cumulative < threshold and remaining.numel() > 0:
        sampled = torch.multinomial(probs[remaining], num_samples=1)
        idx = remaining[sampled].squeeze()

        if not torch.any(selected == idx):
            selected = torch.cat([selected, idx.unsqueeze(0)])
            cumulative += probs[idx].item()
        
        mask = torch.ones_like(remaining, dtype=torch.bool)
        mask[sampled] = False
        remaining = remaining[mask]
    
    if selected.numel() == 0:
        selected = probs.argmax().unsqueeze(0)
    
    return selected

