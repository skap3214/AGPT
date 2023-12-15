'''
All nn.Modules that are implemented and used to create model
'''
import torch
from torch import nn
from typing import Tuple, Optional
import math
from rotary_embedding_torch import RotaryEmbedding

#Very beggining stuff
class Head(nn.Module):
    '''
    My First Scaled Masked Self Attention Head!!!
    '''
    def __init__(
            self, 
            head_size, #Head size to convert the embedding vector into
            n_embd, #Embedding dimension
            block_size, #context length of the model
            dropout, #amoutn of dropout to apply at the end
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, (head_size), bias=False)
        self.query = nn.Linear(n_embd, (head_size), bias=False)
        self.value = nn.Linear(n_embd, (head_size), bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))# basically self.tril = torch.tril(...
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_embd: torch.Tensor):
        B, T, C = token_embd.shape
        #Calculate keys, queries of sizes (B, T, head_size)
        k = self.key(token_embd)
        q = self.query(token_embd)

        #Create scaled masked attention matrix of size (T, T)
        weights = k @ q.transpose(-2, -1) * C**0.5
        weights = weights.masked_fill(self.tril == 0, float('-inf')) #If a value in self.tril == 0 then make the corresponding value in the weights matrix -inf
        weights = torch.softmax(weights, dim=1)
        weights = self.dropout(weights)

        #Multiply weights with values(B, T, head_size) to get size (B, T, head_size)
        v = self.value(token_embd)
        output = weights @ v
        return output


class ScaledHead(nn.Module):
    def __init__(
            self, 
            n_embd,
            block_size,
            dropout,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd*2, bias=False)
        self.query = nn.Linear(n_embd, n_embd*2, bias=False)
        self.value = nn.Linear(n_embd, n_embd*2, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_embd*2, n_embd)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor):

        B, T, C = x.shape
        #Get key, query
        k = self.key(x) #(B, T, 2C)
        q = self.query(x)# (B, T, 2C)

        #Get attention matrix weights
        weights = k @ q.transpose(-2, -1) * C**0.5 #(T, T, 2C)
        weights = weights.masked_fill(self.tril == 0, float("-inf"))
        weights = torch.softmax(weights, dim=1)
        weights = self.dropout(weights)

        #Multiple attention matrix with value
        v = self.value(x) #(B, T, 2C)
        out = weights @ v
        out = self.linear(out) #(B, T, C)
        out = self.relu(out)
        return out

class Block(nn.Module):
    def __init__(
            self, 
            n_embd, 
            num_head,
            block_size,
            dropout,
    ) -> None:
        assert n_embd % num_head == 0, f"not divisible: mod is {n_embd % num_head}"
        super().__init__()
        head_size = n_embd // num_head
        self.mh_head = MultiHeadAttention(head_size, n_embd, block_size, num_head, dropout)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x:torch.Tensor):
        out = x + self.mh_head(self.layer_norm1(x)) #Residual connections
        out = out + self.ffwd(self.layer_norm2(out))
        return out


class FeedForward(nn.Module):
    def __init__(
            self, 
            n_embd,
            dropout=0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    '''
    #TODO: make this and Head as one module
    '''
    def __init__(
            self, 
            head_size,
            n_embd,
            block_size,
            num_heads, 
            dropout
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_embd: torch.Tensor):
        #Concatenate outputs from all heads in the C dimension.
        #Each head corresponds to a different embddding dimension, thus will have differnet weights and stuff
        out = torch.cat([head(token_embd) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


#V2
class MistralMHA(nn.Module):
    def __init__(
        self,
        n_heads,
        n_kv_heads,
        dim,
        head_dim,
        sliding_window,
        max_batch_size = 0,
        dropout = 0
        ):
        super().__init__()

        self.n_heads: int = int(n_heads)
        self.n_kv_heads: int = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.repeats = self.n_heads // self.n_kv_heads #Need to be divisible
        self.sliding_window = int(sliding_window)

        self.scale = head_dim**-0.5

        self.wq = nn.Linear(
            in_features=dim,
            out_features=int(n_heads * head_dim),
            bias=False
        )
        self.wk = nn.Linear(
            in_features=dim,
            out_features=int(n_kv_heads * head_dim),
            bias=False
        )
        self.wv = nn.Linear(
            in_features=dim,
            out_features=int(n_kv_heads * head_dim),
            bias=False
        )
        self.wo = nn.Linear(
            in_features=(int(n_heads * head_dim)),
            out_features=dim,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

        self.cache_k = torch.empty(
            (
                int(max_batch_size),
                int(sliding_window),
                int(self.n_kv_heads),
                int(self.head_dim),
            ), dtype=torch.float16
        )
        self.cache_v = torch.empty(
            (
                int(max_batch_size),
                int(sliding_window),
                int(self.n_kv_heads),
                int(self.head_dim),
            ), dtype=torch.float16
        )

    def apply_rotary_emb(
    self,
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self._reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
        keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
        values = torch.repeat_interleave(values, repeats=repeats, dim=2)
        return keys, values

    def _reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        freqs_cis: complex - (seq_len, head_dim / 2)
        x: complex - (bsz, seq_len, head_dim / 2)
        """
        ndim = x.ndim
        assert 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
            freqs_cis.shape,
            (x.shape[1], x.shape[-1]),
        )
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    
    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # The cache is a rotating buffer
        scatter_pos = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])


        if positions.shape[0] > 1:
            # prefill
            key, value = self.repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = self.repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)
            
        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # scores : [bsz, n_heads, seqlen | 1, seqlen]
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        
        if mask is not None:
            scores += mask[None, None, ...]

        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return self.dropout(output)


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        from: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
        """
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf

    from: https://blog.briankitano.com/llama-from-scratch/
    """
    def __init__(
        self, 
        size
    ):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out


class FeedForwardWithSwiGLU(nn.Module):
    """Feed Forward module with SwiGLU activation"""

    def __init__(
        self,
        n_embd,
        hidden_dim,
        dropout = 0
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            SwiGLU(hidden_dim),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MistralTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        n_embd,
        head_dim,
        sliding_window,
        mistral_mha: MistralMHA,
        device
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.sliding_window = sliding_window
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)

        self.layers = torch.nn.ModuleList(
            [mistral_mha for _ in range(n_layers)]
        )

        self.norm = RMSNorm(n_embd)

        self.output = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        self.freqs_cis = self.precompute_freqs_cis(head_dim, 128_000).to(device)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.sliding_window)
            mask = torch.log(mask)
        
        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask)

        return self.output(self.norm(h)).float()

#V3

class PositionalEncoding(nn.Module):

    def __init__(self, n_embd: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(block_size, 1, n_embd)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_length, embedding_dim]``
        """
        B, T, C = x.shape
        x = x.view((T, B, C))
        x = x + self.pe[:x.size(0)]
        T, B, C = x.shape
        x = x.view((B, T, C)) 
        x = self.dropout(x)
        return x

#V4
class RopeAttention(nn.Module):

    def __init__(
            self,
            n_embd,
            dropout,
            block_size,
            n_head,
            bias=True,
        ):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.rope = RotaryEmbedding(dim=(n_embd//n_head))
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x, pad_mask=None):
        B, T, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        # Combine with pad_mask if it's provided
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)  # Reshape to (B, 1, 1, T)
            combined_mask = causal_mask * pad_mask
        else:
            combined_mask = causal_mask
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=combined_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if combined_mask is not None:
                att = att.masked_fill(combined_mask == 0, float('-inf'))
            else:
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class RopeBlock(nn.Module):
    def __init__(self, n_embd, num_head, block_size, dropout):
        assert n_embd % num_head == 0, f"not divisible: mod is {n_embd % num_head}"
        super().__init__()
        self.mh_head = RopeAttention(n_embd, dropout, block_size, num_head)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x: torch.Tensor, pad_mask=None):
        out = x + self.mh_head(self.layer_norm1(x), pad_mask=pad_mask)  # Residual connections
        out = out + self.ffwd(self.layer_norm2(out))
        return out

class RopeWithRMSNormBlock(nn.Module):
    def __init__(
            self, 
            n_embd,
            num_head,
            block_size,
            dropout,
    ) -> None:
        assert n_embd % num_head == 0, f"not divisible: mod is {n_embd % num_head}"
        super().__init__()
        self.mh_head = RopeAttention(n_embd, dropout, block_size, num_head)
        self.ffwd = FeedForward(n_embd)
        self.rms_norm = RMSNorm(n_embd)
    
    def forward(self, x:torch.Tensor, pad_mask=None):
        out = x + self.mh_head(self.rms_norm(x), pad_mask=pad_mask) #Residual connections
        out = out + self.ffwd(self.rms_norm(out))
        return out