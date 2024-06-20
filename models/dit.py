import torch
import torch.nn.functional as F
import huggingface_hub
from .modules.transformer import Transformer
from .modules.embeddings.time import TimestepEmbedder

class DIT(Transformer, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__(config, vocab_size, adaptive=True)
    assert self.causal == False
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)

  def forward(self, indices, sigma):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, seqlens=None, c=c)
      x = self.output_layer(x, c)

    return x