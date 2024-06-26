from .sedd import SEDD
from .d3pm import D3PM
from .mdlm import MDLM
from .ar import AR

def get_diffusion(config, tokenizer):
  Diffusion = _get_diffusion_class(config.parameterization)
  return Diffusion(config, tokenizer)

def get_diffusion_from_checkpoint(checkpoint_path, config, tokenizer):
  Diffusion = _get_diffusion_class(config.parameterization)
  return Diffusion.load_from_checkpoint(checkpoint_path, config, tokenizer)
  
def _get_diffusion_class(class_name):
  if class_name == 'sedd':
    return SEDD
  elif class_name == 'd3pm':
    return D3PM
  elif class_name == 'subs':
    return MDLM
  elif class_name == 'ar':
    return AR
  else:
    raise NotImplementedError()    