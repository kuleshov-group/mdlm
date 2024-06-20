import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn
from einops import rearrange

from .embeddings.vocab import EmbeddingLayer
from .embeddings.rotary import Rotary, apply_rotary_pos_emb
from .normalization import LayerNorm
from .scaling import (
  bias_dropout_add_scale_fused_train, 
  bias_dropout_add_scale_fused_inference,
  modulate_fused
)

class Transformer(nn.Module):
  def __init__(self, config, vocab_size: int, adaptive=True):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = (
      hasattr(config.model, 'causal')
      and config.model.causal
    )

    self.vocab_embed = EmbeddingLayer(
      config.model.hidden_size, vocab_size
    )
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads
    )

    # choose layer times
    Block, FinalLayer = (
      (AdaTransformerBlock, AdaTransformerFinalLayer) if adaptive 
      else (TransformerBlock, TransformerFinalLayer)
    )

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(
        Block(
          config.model.hidden_size,
          config.model.n_heads,
          config.model.cond_dim,
          dropout=config.model.dropout,
          causal=self.causal,
        )
      )
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = FinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim,
      causal=self.causal,
    )

class TransformerBlock(nn.Module):
  def __init__(
    self,
    dim,
    n_heads,
    cond_dim,
    mlp_ratio=4,
    dropout=0.1,
    causal=False,
  ):
    super().__init__()
    self.n_heads = n_heads
    self.causal = causal

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True),
    )
    self.dropout = dropout

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def _flashattention(self, x, rotary_cos_sin, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]
    qkv = self.attn_qkv(x)
    qkv = rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads,
    )
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
      )
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    if seqlens is None:
      cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        step=seq_len,
        dtype=torch.int32,
        device=qkv.device,
      )
    else:
      cu_seqlens = seqlens.cumsum(-1)
    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      qkv, cu_seqlens, seq_len, 0.0, causal=self.causal
    )

    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    return x

  def forward(self, x, rotary_cos_sin, seqlens=None):
    x_skip = x
    x = self.norm1(x)
    x = self._flashattention(x, rotary_cos_sin, seqlens)

    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    scale = torch.ones(1, device=x.device, dtype=x.dtype)
    x = bias_dropout_scale_fn(
      self.attn_out(x), None, scale, x_skip, self.dropout
    )

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(self.norm2(x)), None, scale, x, self.dropout
    )
    return x

class AdaTransformerBlock(TransformerBlock):
  def __init__(
    self,
    dim,
    n_heads,
    cond_dim,
    mlp_ratio=4,
    dropout=0.1,
    causal=False,
  ):
    super().__init__(dim, n_heads, cond_dim, mlp_ratio, dropout, causal)
    self.adaLN_modulation = nn.Linear(cond_dim, 6*dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _modulated_forward(self, x, rotary_cos_sin, c, seqlens=None):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp \
        = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    x = self._flashattention(x, rotary_cos_sin, seqlens)

    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x

  def forward(self, x, rotary_cos_sin, seqlens=None, c=None):
    if c is not None:
      return self._modulated_forward(x, rotary_cos_sin, c, seqlens)
    else:
      raise NotImplementedError() # or call superclass

class TransformerFinalLayer(nn.Module):
  def __init__(
    self, hidden_size, out_channels, cond_dim, causal=False
  ):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.causal = causal

  def forward(self, x):
      return self.linear(self.norm_final(x))

class AdaTransformerFinalLayer(TransformerFinalLayer):
  def __init__(
    self, hidden_size, out_channels, cond_dim, causal=False
  ):
    super().__init__(hidden_size, out_channels, cond_dim, causal)
    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    return self.linear(x)