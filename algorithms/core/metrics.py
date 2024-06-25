import math
import torch
import torchmetrics
from dataclasses import dataclass
from torch import Tensor

LOG2 = math.log(2)

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass

class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2

class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)