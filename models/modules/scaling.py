import typing
import torch
import torch.nn.functional as F

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

# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)