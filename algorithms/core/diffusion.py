import itertools
import numpy as np
import torch
import transformers

import models
import noise_schedule
import utils


class CoreDiffusion(torch.nn.Module):
  def __init__(
    self,
    config,
    vocab_size: int,
    mask_token_id=None,
    pad_token_id=None
    # tokenizer: transformers.PreTrainedTokenizer
  ):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    # FIXE: TODO: tokenizer should not be here
    # worst case, it should be in LightningDiffusion (also not ideal)
    # self.tokenizer = tokenizer 
    # self.vocab_size = self.tokenizer.vocab_size
    self.vocab_size = self.vocab_size
    self.sampler = self.config.sampling.predictor
    
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    
    # save mask token id
    # if (not hasattr(self.tokenizer, 'mask_token')
    #     or self.tokenizer.mask_token is None):
    if mask_token_id is None:
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = mask_token_id
    self.parameterization = self.config.parameterization

    # set backbone
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size
      )
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=pad_token_id
      )
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index
      )
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.noise = noise_schedule.get_noise(
      self.config, dtype=self.dtype
    )
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self.iter_params(),
        decay=self.config.training.ema
      )
    else:
      self.ema = None

    self.T = self.config.T    
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self._validate_configuration()

  # main abstract methods that need to be implemented by subclasses:

  def forward(self, x, sigma):
    raise NotImplementedError()

  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    raise NotImplementedError()

  def _diffusion_elbo(self, x0):
    raise NotImplementedError()

  # core diffusion methods:

  def _loss(self, x0, attention_mask):
    input_tokens, token_mask = x0, attention_mask

    elbo = self._diffusion_elbo(input_tokens)
    nlls = elbo * token_mask
    count = token_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=token_mask)

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  # verificaiton methods

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  # methods that set train vs. eval mode

  def train_mode(self, ema=self.ema):
    """Set model to train mode"""
    if ema:
      self.restore_ema()
    self.backbone.train()
    self.noise.train()

  def eval_mode(self, ema=self.ema):
    """Set model to eval mode"""
    if ema:
      self.store_ema()
    self.backbone.eval()
    self.noise.eval()

  def iter_params(self):
    return itertools.chain(
      self.backbone.parameters(),
      self.noise.parameters()
    )

    logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]

  # abstract interface over EMA

  def store_ema(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def restore_ema(self):
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def update_ema(self):
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def load_ema_from_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])

  def save_ema_to_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    return checkpoint

  def move_ema_shadow_params_to_device():
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)

  # helper methods (to be refactored):

  # TODO: FIXME: these funcs should also not be here
  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    self.eval_mode()
    samples = self._sample(num_steps=num_steps, eps=eps)
    self.train_mode()
    return samples

  # TODO: FIXME: this should be merged with regular sampling functions
  # should also implement semi-AR MDLM algorithm
  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    self.eval_mode()
    sampling_steps, samples, sequence_lengths = self.semi_ar_sample(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt
    )
    self.train_mode()
    return sampling_steps, samples, sequence_lengths  