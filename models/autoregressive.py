import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
  x: torch.Tensor,
  bias: typing.Optional[torch.Tensor],
  scale: torch.Tensor,
  residual: typing.Optional[torch.Tensor],
  prob: float,
  training: bool,
) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(
      x + bias, p=prob, training=training
    )
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training
    )

  return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_scale_fused_train(
  x: torch.Tensor,
  bias: typing.Optional[torch.Tensor],
  scale: torch.Tensor,
  residual: typing.Optional[torch.Tensor],
  prob: float,
) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True
  )


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
  x: torch.Tensor,
  bias: typing.Optional[torch.Tensor],
  scale: torch.Tensor,
  residual: typing.Optional[torch.Tensor],
  prob: float,
) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False
  )


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (
      base ** (torch.arange(0, dim, 2).float() / dim)
    )
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(
        x.shape[seq_dim], device=x.device
      ).type_as(self.inv_freq)
      freqs = torch.einsum(
        'i,j->ij', t, self.inv_freq.clone()
      )
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[
        None, :, None, None, :
      ].repeat(1, 1, 3, 1, 1)
      self.sin_cached = emb.sin()[
        None, :, None, None, :
      ].repeat(1, 1, 3, 1, 1)
      # This makes the transformation on v an identity.
      self.cos_cached[:, :, 2, :, :].fill_(1.0)
      self.sin_cached[:, :, 2, :, :].fill_(0.0)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = (
    x[..., : x.shape[-1] // 2],
    x[..., x.shape[-1] // 2 :],
  )
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
  sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(
    qkv, cos, sin
  )


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim

  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale,
  ).view(*x.shape[:-1], dim_out)


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
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
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True),
    )
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # attention operation
    x_skip = x
    x = self.norm1(x)

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

    scale = torch.ones(1, device=x.device, dtype=x.dtype)
    x = bias_dropout_scale_fn(
      self.attn_out(x), None, scale, x_skip, self.dropout
    )

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(self.norm2(x)), None, scale, x, self.dropout
    )
    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(
      torch.empty((vocab_dim, dim))
    )
    torch.nn.init.kaiming_uniform_(
      self.embedding, a=math.sqrt(5)
    )

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(
    self, hidden_size, out_channels, cond_dim, causal=False
  ):
    super().__init__()
    self.causal = causal
    assert causal == True

    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

  def forward(self, x, c):
    return self.linear(self.norm_final(x))


class DDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = (
      hasattr(config.model, 'causal')
      and config.model.causal
    )
    assert self.causal == True

    self.vocab_embed = EmbeddingLayer(
      config.model.hidden_size, vocab_size
    )
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads
    )

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(
        DDiTBlock(
          config.model.hidden_size,
          config.model.n_heads,
          config.model.cond_dim,
          dropout=config.model.dropout,
          causal=self.causal,
        )
      )
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim,
      causal=self.causal,
    )
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


class AR(DDIT):
  def __init__(self, config, vocab_size, mask_index):
    super().__init__(config, vocab_size)
    self.mask_index = mask_index
    self.neg_infinity = -1000.0

  def forward(self, xt, sigma):
    """Forward pass of the denoising model.

    Args:
      xt: int torch.Tensor with shape
          (batch_size, diffusion_model_input_length), token ids.
      sigma: float torch.Tensor with shape
          (batch_size).

    Returns:
      log probability with shape
          (batch_size, diffusion_model_input_length, vocab_size)
    """
    x = self.vocab_embed(xt)

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary_cos_sin, None, seqlens=None
        )
      output = self.output_layer(x, None)

    # log prob at the mask index = - infinity
    output[:, :, self.mask_index] = self.neg_infinity

    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    # x = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return output.log_softmax(-1)
