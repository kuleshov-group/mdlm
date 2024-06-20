import torch
import huggingface_hub
from .modules.transformer import Transformer

class AR(Transformer, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size, mask_index):
    super().__init__(config, vocab_size, adaptive=False)
    assert self.causal == True
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
          x, rotary_cos_sin, seqlens=None
        )
      output = self.output_layer(x)

    # log prob at the mask index = - infinity
    output[:, :, self.mask_index] = self.neg_infinity

    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    # x = x - torch.logsumexp(x, dim=-1, keepdim=True)
    return output.log_softmax(-1)
