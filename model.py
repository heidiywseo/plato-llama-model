from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    def __init__(self, dim=256, n_layers=4, n_heads=8, n_kv_heads=None, vocab_size=-1, multiple_of=256,
                 ffn_dim_multiplier=None, norm_eps=1e-5, max_batch_size=32, max_seq_len=2048, device=None):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = None
        self.vocab_size = vocab_size  # Later set in the build method
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.device = device



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :] # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)

        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)

        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        xq = self.wq(x) # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xk = self.wk(x) # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim))
        xv = self.wv(x) # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim) # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device) # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device) # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)

        self.cache_k = self.cache_k.clone() # Create a clone to avoid in-place modification
        self.cache_v = self.cache_v.clone()

        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk.detach()
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv.detach()

        keys = self.cache_k[:batch_size, : start_pos + seq_len] # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len] # (B, Seq_Len_KV, H_KV, Head_Dim)
        
        keys = repeat_kv(keys, self.n_rep) # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep) # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)

        xq = xq.transpose(1, 2) # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        keys = keys.transpose(1, 2) # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2) # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)

        output = torch.matmul(scores, values) # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)) # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x)) # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x) # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = self.w2(x) # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        h_ffn = self.feed_forward.forward(self.ffn_norm(h))
        out = h + h_ffn
        return out
    
class llamaModel(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        assert isinstance(args.vocab_size, int), f"Expected int for vocab_size but got {type(args.vocab_size)}"
        assert isinstance(args.dim, int), f"Expected int for dim but got {type(args.dim)}"

        self.args = args
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.vocab_size = args.vocab_size  
        self.multiple_of = args.multiple_of
        self.ffn_dim_multiplier = args.ffn_dim_multiplier
        self.norm_eps = args.norm_eps
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.device = args.device

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output