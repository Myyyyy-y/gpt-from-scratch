"""Decoder-only transformer language model (LLaMA-style), built from scratch.

Implements RoPE, RMSNorm/LayerNorm, SwiGLU/GELU/ReLU^2, causal attention with
KV-cache support, and ablation switches (QK-Norm, zero-init output projections,
Attention Residuals). Only nn.Module and basic tensor ops are used.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """All hyperparameters; ablations toggle one field at a time."""

    vocab_size: int = 8192
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1344
    context_length: int = 256
    dropout: float = 0.0
    tie_weights: bool = True
    # --- ablation switches ---
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    ffn_type: str = "swiglu"    # "swiglu" | "gelu" | "relu2"
    pos_type: str = "rope"      # "rope" | "learned" | "none"
    # --- training-technique switches (default off for checkpoint compat) ---
    qk_norm: bool = False
    zero_init_proj: bool = False
    attn_res: bool = False


# ---------- normalization ----------

class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.weight


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * weight + bias"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return (y.to(dtype)) * self.weight + self.bias


def make_norm(norm_type, dim):
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return LayerNorm(dim)
    raise ValueError(f"未知 norm_type: {norm_type}")


# ---------- softmax ----------

def softmax(x, dim=-1):
    """Numerically stable softmax (subtract max before exp)."""
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


# ---------- RoPE ----------

def precompute_rope(context_length, head_dim, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(context_length, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


# ---------- causal attention ----------

def causal_attention(q, k, v):
    """Scaled dot-product attention with a causal mask. Supports T != S (KV cache)."""
    D = q.shape[-1]
    T, S = q.shape[-2], k.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    # diagonal offset unifies prefill (T=S) and decode (T=1) masking
    mask = torch.triu(torch.ones(T, S, device=q.device, dtype=torch.bool),
                      diagonal=1 + (S - T))
    scores = scores.masked_fill(mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_rope = (config.pos_type == "rope")
        self.use_qk_norm = config.qk_norm
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        if self.use_qk_norm:
            # normalize Q/K before scoring to keep logits scale bounded
            # (standard in Gemma3/Qwen3/OLMo2)
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        if self.use_rope:
            # precompute beyond context_length so generation can exceed training length
            max_len = max(config.context_length, 1024)
            cos, sin = precompute_rope(max_len, self.head_dim)
            # persistent=False: constant tables, not part of the state dict
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, positions=None, past_kv=None, use_cache=False):
        """Returns (out, (K, V)) when use_cache=True, else out."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            if positions is None:
                past_len = past_kv[0].shape[2] if past_kv is not None else 0
                positions = torch.arange(past_len, past_len + T, device=x.device)
            cos = self.cos[positions].to(dtype=x.dtype)
            sin = self.sin[positions].to(dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if past_kv is not None:
            # rotate new K first, then concat: cached history is already rotated
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        y = causal_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(self.dropout(y))
        if use_cache:
            return out, (k, v)
        return out


# ---------- FFN variants ----------

class SwiGLU(nn.Module):
    """w2(SiLU(w1(x)) * w3(x)) — gated FFN (LLaMA)."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeluMLP(nn.Module):
    """w2(GELU(w1(x))) — classic MLP (GPT-2), exact GELU via erf."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        h = self.w1(x)
        h = 0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))
        return self.w2(h)


class ReluSquaredMLP(nn.Module):
    """w2(relu(w1(x))^2) — cheaper than GELU, comparable quality."""

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        h = torch.relu(self.w1(x)) ** 2
        return self.w2(h)


def make_ffn(config):
    if config.ffn_type == "swiglu":
        return SwiGLU(config)
    if config.ffn_type == "gelu":
        return GeluMLP(config)
    if config.ffn_type == "relu2":
        return ReluSquaredMLP(config)
    raise ValueError(f"未知 ffn_type: {config.ffn_type}")


# ---------- transformer block ----------

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = make_norm(config.norm_type, config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = make_norm(config.norm_type, config.d_model)
        self.ffn = make_ffn(config)

    def forward(self, x, positions=None, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.attn_norm(x), positions=positions,
                                     past_kv=past_kv, use_cache=True)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, (new_kv if use_cache else None)


# ---------- language model ----------

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.pos_type == "learned":
            self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = make_norm(config.norm_type, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.attn_res:
            # deep residual routing: each layer's input is a softmax-weighted
            # combination of all previous layer outputs (Kimi 2024)
            self.attn_res_queries = nn.ParameterList(
                [nn.Parameter(torch.zeros(config.d_model)) for _ in range(config.n_layers)])
            self.attn_res_norm = RMSNorm(config.d_model)

        self.apply(self._init_weights)

        if config.zero_init_proj:
            # start each block as identity: sublayer output is 0 at init
            for blk in self.blocks:
                nn.init.zeros_(blk.attn.out_proj.weight)
                nn.init.zeros_(blk.ffn.w2.weight)

        if config.tie_weights:
            # must tie after _init_weights to avoid double init
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            std = math.sqrt(2.0 / (module.weight.shape[0] + module.weight.shape[1]))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, nn.Embedding):
            # with tied weights E doubles as lm_head; std=1 would blow up initial
            # logits (loss >> ln(vocab)), so use GPT-2-style std=0.02
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(self, idx, positions=None, past_kvs=None, use_cache=False,
                return_hidden_states=False):
        """idx: (B, T) token ids -> logits (B, T, vocab_size), plus caches if requested."""
        B, T = idx.shape
        if past_kvs is None:
            assert T <= self.config.context_length, f"序列长度 {T} 超过 context_length"

        if positions is None:
            past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
            positions = torch.arange(past_len, past_len + T, device=idx.device)

        x = self.token_embedding(idx)

        if self.config.pos_type == "learned":
            x = x + self.position_embedding(positions)

        if self.config.attn_res:
            assert not use_cache, "attn_res 模式不支持 KV cache（请用 use_cache=False）"
            hidden_states = [x]
            for i, block in enumerate(self.blocks):
                if i == 0:
                    h_in = x
                else:
                    Q = self.attn_res_queries[i]
                    V = torch.stack(hidden_states)
                    K = self.attn_res_norm(V)
                    scores = torch.einsum("lbsd,d->lbs", K, Q)
                    scores = scores / math.sqrt(self.config.d_model)
                    attn = torch.softmax(scores, dim=0)
                    h_in = torch.einsum("lbsd,lbs->bsd", V, attn)
                h_out, _ = block(h_in, positions=positions)
                hidden_states.append(h_out)
            x = hidden_states[-1]
            x = self.norm_f(x)
            logits = self.lm_head(x)
            if return_hidden_states:
                return logits, hidden_states
            return logits

        new_kvs = []
        hidden_states = [x] if return_hidden_states else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, positions=positions, past_kv=past_kv, use_cache=use_cache)
            if hidden_states is not None:
                hidden_states.append(x)
            new_kvs.append(new_kv)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if use_cache:
            return logits, new_kvs
        if return_hidden_states:
            return logits, hidden_states
        return logits
