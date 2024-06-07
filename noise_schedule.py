import abc

import torch
import torch.nn as nn

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_noise(config, dtype=torch.float32):
  if config.noise.type == 'geometric':
    return GeometricNoise(config.noise.sigma_min,
                          config.noise.sigma_max)
  elif config.noise.type == 'loglinear':
    return LogLinearNoise()
  elif config.noise.type == 'cosine':
    return CosineNoise()
  elif config.noise.type == 'cosinesqr':
    return CosineSqrNoise()
  elif config.noise.type == 'linear':
    return Linear(config.noise.sigma_min,
                  config.noise.sigma_max,
                  dtype)
  else:
    raise ValueError(f'{config.noise.type} is not a valid noise')


def binary_discretization(z):
  z_hard = torch.sign(z)
  z_soft = z / torch.norm(z, dim=-1, keepdim=True)
  return z_soft + (z_hard - z_soft).detach()


class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  def forward(self, t):
    # Assume time goes from 0 to 1
    return self.total_noise(t), self.rate_noise(t)
  
  @abc.abstractmethod
  def rate_noise(self, t):
    """
    Rate of change of noise ie g(t)
    """
    pass

  @abc.abstractmethod
  def total_noise(self, t):
    """
    Total noise ie \int_0^t g(t) dt + g(0)
    """
    pass


class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2)
    return - torch.log(self.eps + (1 - self.eps) * cos)


class CosineSqrNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def rate_noise(self, t):
    cos = (1 - self.eps) * (
      torch.cos(t * torch.pi / 2) ** 2)
    sin = (1 - self.eps) * torch.sin(t * torch.pi)
    scale = torch.pi / 2
    return scale * sin / (cos + self.eps)

  def total_noise(self, t):
    cos = torch.cos(t * torch.pi / 2) ** 2
    return - torch.log(self.eps + (1 - self.eps) * cos)


class Linear(Noise):
  def __init__(self, sigma_min=0, sigma_max=10, dtype=torch.float32):
    super().__init__()
    self.sigma_min = torch.tensor(sigma_min, dtype=dtype)
    self.sigma_max = torch.tensor(sigma_max, dtype=dtype)

  def rate_noise(self, t):
    return self.sigma_max - self.sigma_min

  def total_noise(self, t):
    return self.sigma_min + t * (self.sigma_max - self.sigma_min)

  def importance_sampling_transformation(self, t):
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    return (sigma_t - self.sigma_min) / (
      self.sigma_max - self.sigma_min)


class GeometricNoise(Noise):
  def __init__(self, sigma_min=1e-3, sigma_max=1):
    super().__init__()
    self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

  def rate_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (
      self.sigmas[1].log() - self.sigmas[0].log())

  def total_noise(self, t):
    return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def importance_sampling_transformation(self, t):
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    t = - torch.expm1(- sigma_t) / (1 - self.eps)
    return t
