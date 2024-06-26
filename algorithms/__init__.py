from .sedd import SEDD
from .d3pm import D3PM
from .mdlm import MDLM
from .ar import AR

def get_diffusion(config, tokenizer):
  if config.parameterization == 'sedd':
    return SEDD(config, tokenizer)
  elif config.parameterization == 'd3pm':
    return D3PM(config, tokenizer)
  elif config.parameterization == 'mdlm':
    return MDLM(config, tokenizer)
  elif config.parameterization == 'ar':
    return AR(config, tokenizer)
  else:
    raise NotImplementedError()