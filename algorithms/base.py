"""Implements hooks used by PyTorch Lightning"""
import abc
import torch
import torchmetrics
import transformers
import dataloader
import lightning as L
from .core.diffusion import CoreDiffusion
from .core.metrics import NLL, BPD, Perplexity

active_metrics = {
  'nll': NLL(),
  'bpd': BPD(),
  'ppl': Perplexity(),
}

class DiffusionAlgorithm(CoreDiffusion, abc.ABC):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer
  ):
    CoreDiffusion.__init__(
      self, 
      config, 
      vocab_size=tokenizer.vocab_size,
      mask_token_id=tokenizer.mask_token_id,
      pad_token_id=tokenizer.pad_token_id
    )
    self.config = config
    self.tokenizer = tokenizer #TODO: tokenizer should be in its own callback
    self.ema_status = (self.config.training.ema > 0)

    self.fast_forward_epochs = None
    self.fast_forward_batches = None

    # if not hasattr(self, 'noise'):
    #     raise NotImplementedError("Subclasses must define 'noise'.")
    # if not hasattr(self, 'compute_generative_perplexity'):
    #     raise NotImplementedError("Subclasses must define a gen ppl func.")

    # metrics will automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection(active_metrics)
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

  # we want to implement the following abstract functions

  @abc.abstractmethod
  def forward(self, x, sigma):
    raise NotImplementedError()

  @abc.abstractmethod
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _diffusion_elbo(self, x0):
    raise NotImplementedError()

  # extra hooks used by the algorithm

  def on_train_start(self):
    self.move_ema_shadow_params_to_device()
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def on_train_epoch_start(self):
    self.train_mode(ema=False)

  def on_validation_epoch_start(self):
    self.eval_mode(ema=self.ema_status)
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def on_load_checkpoint(self, checkpoint):
    self.load_ema_from_checkpoint()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    checkpoint = self.save_ema_to_checkpoint(checkpoint)
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.config.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    self.restore_ema()
