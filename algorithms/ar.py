import torch
import transformers

from .core.diffusion import CoreDiffusion
from .core.lightning import LightningDiffusion
from .core.genppl import GenPPLEvaluator


class AR(
  CoreDiffusion, LightningDiffusion, GenPPLEvaluator
):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer
  ):
    CoreDiffusion.__init__(
      self, 
      config, 
      mask_token_id=tokenizer.mask_token_id
    )
    LightningDiffusion.__init__(self, config, tokenizer)
    GenPPLEvaluator.__init__(self, config)

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    return logits

  def _loss(self, x0, attention_mask):
    input_tokens, output_tokens, token_mask = (
      self._get_token_mask(x0, attention_mask)
    )
    logprobs = self.backbone(input_tokens, None)
    loss = - logprobs.gather(
      -1, output_tokens[:, :, None])[:, :, 0]
    
    nlls = loss * token_mask
    count = token_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=token_mask)

  def _get_token_mask(self, x0, attention_mask):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    new_attention_mask = attention_mask[:, 1:]

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    return self._ar_sampler(batch_size_per_gpu)
    