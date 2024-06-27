"""Implements training hooks for PyTorch Lightning"""
import abc
import torch
import hydra.utils
import lightning as L

class CoreLightning(L.LightningModule, abc.ABC):
  def __init__(
    self,
    config,
  ):
    L.LightningModule.__init__(self)
    self.config = config
    self.ema_status = (self.config.training.ema > 0)

  # we want to implement the following abstract functions

  @abc.abstractmethod
  def forward(self, x, sigma_t):
    raise NotImplementedError()

  @abc.abstractmethod
  def _loss(batch, attention_mask):
    raise NotImplementedError()

  def optimizer_step(self, *args, **kwargs):
    L.LightningModule.optimizer_step(self, *args, **kwargs)
    self.update_ema()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self.iter_params(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

