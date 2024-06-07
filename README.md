# Text Diffusion
We provide efficient implementations for numerous baselines
SEDD, D3PM (absorbing state), MDLM (ours). 

## Generate Samples
Using the huggingface model:
```
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=hf_dit
```
For a local checkpoint
```
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=10000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=dit
```

### Semi-ar samples
```
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  parameterization=subs \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  sampling.semi_ar=True \
  sampling.stride_length=512 \
  sampling.num_strides=2 \
  backbone=hf_dit
```


## Train
### MDLM
```
python -u -m main \
  trainer.max_steps=1000000 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000
```
### SEDD
```
python -u -m main \
  trainer.max_steps=1000000 \
  model=small \
  data=openwebtext-split \
  wandb.name=sedd-owt \
  parameterization=sedd \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  time_conditioning=True
```

### D3PM
```
python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.max_steps=1000000 \
  model=small \
  data=lm1b \
  wandb.name=d3pm-lm1b \
  parameterization=d3pm \
  model.length=128 \
  sampling.steps=1000 \
  T=1000 \
  time_conditioning=True
```
### AR
```
python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.max_steps=1000000 \
  model=small \
  data=lm1b \
  wandb.name=ar-lm1b \
  parameterization=ar \
  model.length=128 \
  checkpointing.save_dir=${SAVE_CKPT_DIR} \
  eval.compute_generative_perplexity=True \
  sampling.steps=1024 \
  backbone=ar
```

## Eval 
To evaluate a checkpoint do this
### Perplexity
```
python main.py mode=ppl_eval eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-mulan-v2-scalar-owt-not-tZycWP-small-param-subs_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt data=openwebtext-split parameterization=subs   model.length=1024  sampling.predictor=ddpm_cache   time_conditioning=False sampling.steps=1000 loader.eval_batch_size=4 
```

