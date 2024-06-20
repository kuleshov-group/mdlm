import math
import torch
import torch.nn as nn


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