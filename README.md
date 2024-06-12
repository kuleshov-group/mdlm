# [Simple and Effective Masked Diffusion Language Models](http://arxiv.org/abs/2406.07524)
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Marianne Arriola](https://mariannearriola.github.io), [Yair Schiff](https://yair-schiff.github.io), [Aaron Gokaslan](https://skylion007.github.io), [Edgar Marroquin](https://emarro.github.io),
[Justin T Chiu](https://justinchiu.netlify.app), [Alexander Rush](https://rush-nlp.com), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nC6q7dWq154fI1BXPLwmtnS7Zvbrv6p?usp=sharing/)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://github.com/huggingface/datasets-tagging/actions/workflows/deploy.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2406.07524)
[![deploy](https://img.shields.io/badge/Huggingface%20-MDLM%20-blue)](https://huggingface.co/kuleshov-group/mdlm-owt)

![graphical_abstract_updated_2](https://github.com/s-sahoo/mdlm/assets/16799748/b0cab23a-d966-45fa-a3ad-be972b23a98a)

We introduce MDLM, a masked discrete diffusion model that features
a novel (SUBS)titution based
parameterization which simplifies the absorbing state diffusion
loss to a mixture of
classical masked language modeling losses. In doing so, we achieve
SOTA perplexity numbers on LM1B and OpenWebText among diffusion models while achiving competitive zero-shot perplexity with SOTA AR models on numerous datasets. We provide a demo in this [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nC6q7dWq154fI1BXPLwmtnS7Zvbrv6p?usp=sharing/) notebook.


In this repo, we release:
* The MDLM framework.
* Implementations of numerous baselines [Examples](#baselines):
  * Autoregressive model that matches the SOTA AR performance on LM1B.
  * Score Entropy Based Discrete Diffusion [SEDD](https://arxiv.org/abs/2310.16834).
  * An efficient implementation of the absorbing state [D3PM](https://arxiv.org/abs/2107.03006) that beats the previous state of the art diffuision model SEDD on LM1B.

* Numerous samplers.
  * Ancestral sampling as proposed in D3PM.
  * Analytic sampler as proposed in SEDD.
  * Our proposed efficient sampler that
    * makes MDLM **~3-4x** faster than the existing diffusion models. [[Example]](#sample-gen)
    * supports semi-autoregressive (SAR) generation.  [[Example]](#semi-ar-gen)


<a name="getting_started"></a>

## Getting started in this repository

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f requirements.yaml
```

Activate the environment.

```bash
conda activate mdlm
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```
and run the training as a batch job:
```bash
sbatch scripts/train_owt_sedd.sh
```

### Checkpoints

We have uploaded MDLM model trained on OpenWebText for 1M training steps to the Huggingface hub 🤗:
[kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt)
Furthermore, we have released the checkpoints for the AR and SEDD baselines trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing).

## Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.
Throughout, the main entry point for running experiments is the [`main.py`](./main.py) script.
We also provide sample `slurm` scripts for launching pre-training and downstream fine-tuning experiments in the [`scrips/`](./scripts) directory.


### Generate Samples
<a name="sample-gen"></a>
The argument to `sampling.predictor` specifies the sampler which takes one of the following values:
* `ddpm_cache`: our proposed sampler that's **~3-4x** faster than the samplers propsed in D3PM and SEDD.
* `ddpm`: Ancestral sampling proposed in D3PM.
* `analytic`: Anlytic sampler proposed in SEDD.

In the following table we report Wall clock time to generate 64 samples on a single A5000 GPU with `batch_size=1`. $T$ denotes the time discretization of the reverse process.
|                         | $T=5k (\downarrow)$ | $T=10k (\downarrow)$ |
|-------------------------|---------------------|----------------------|
| **SEDD**                | 127.1                | 229.3                |
| **MDLM** + `ddpm`            | 206.6                | 229.3                |
| **MDLM** +`ddpm_cache` | **40.1**           | **60.4**             |


To generate samples from a pre-trained model use one of the following commands:
#### Huggingface model
```bash
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
#### Local checkpoint
```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=10000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=dit
```

### Semi-ar sample generation
<a name="semi-ar"></a>
MDLM can also generate samples in a Semi-ar manner.
We generate 200 sequences of length 2048 tokens on a single `3090` GPU and evaluate generative perplexity under a pre-trained GPT-2 model. In the below table we find that in addition to achieving better generative perplexity, MDLM enables **25-30x** faster SAR decoding relative to [SSD-LM](https://arxiv.org/abs/2210.17432).

|                     | Gen. PPL ($\downarrow$) | Sec/Seq ($\downarrow$) |
|---------------------|-------------------------|------------------------|
| **SSD-LM**          | 35.43                   | 2473.9                 |
| **MDLM** +`ddpm_cache`  | **27.18**               | **89.3**               |

*Gen. PPL: Generation Perplexity, Sec/Seq: Seconds per Sequence*

```bash
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


### Train
To train MDLM from scratch on OpenWebText use the following command:
```
python main.py \
  trainer.max_steps=1000000 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000
```
The arguments `loader.batch_size` and `loader.eval_batch_size` allow you to control the batch size per GPU. If `loader.batch_size * num_gpus` is less than the global_batch_size, PyTorch Lightning will resort to gradient accumulation. You can also launch a training job on Slurm using the command: `sbatch scripts/train_owt_mdlm.sh`. The slurm scripts to train the Auto-regressive and SEDD baselines are as follows respectively: [`scripts/train_lm1b_ar.sh`](scripts/train_lm1b_ar.sh), [`scripts/train_owt_sedd.sh`](scripts/train_owt_sedd.sh).

### Eval 
### MDLM
```
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=subs \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  +wandb.offline=true
```

### Baselines
<a name="baselines"></a>
We release the checkpoints for the baselines: SEDD and AR trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing). Download the checkpoints: `ar.ckpt`, `sedd.ckpt` and use the following commands to compute perplexity on the test-split:
#### AR
```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small-ar \
  parameterization=ar \
  backbone=ar \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/ar.ckpt \
  +wandb.offline=true
```
#### SEDD
```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=sedd \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/sedd.ckpt \
  time_conditioning=True \
  +wandb.offline=true
```

### Acknowledgements
This repository was built off of [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).