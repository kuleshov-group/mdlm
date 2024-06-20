# [Simple and Effective Masked Diffusion Language Models](http://arxiv.org/abs/2406.07524)
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Marianne Arriola](https://mariannearriola.github.io), [Yair Schiff](https://yair-schiff.github.io), [Aaron Gokaslan](https://skylion007.github.io), [Edgar Marroquin](https://emarro.github.io),
[Justin T Chiu](https://justinchiu.netlify.app), [Alexander Rush](https://rush-nlp.com), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nC6q7dWq154fI1BXPLwmtnS7Zvbrv6p?usp=sharing/)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://s-sahoo.com/mdlm/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2406.07524)
[![deploy](https://img.shields.io/badge/Huggingface%20-MDLM%20-blue)](https://huggingface.co/kuleshov-group/mdlm-owt)

![graphical_abstract_updated_2](https://github.com/s-sahoo/mdlm/assets/16799748/b0cab23a-d966-45fa-a3ad-be972b23a98a)

This is an experimental fork of the main MDLM [repo](https://github.com/kuleshov-group/mdlm/). This code is experimental, may be broken, and is being actively hacked on as a personal experiment. Please use the official repo for anything serious. 

<a name="code-organization"></a>
## Code Organization
1. ```main.py```: Routines for training and evaluation
2. ```noise_schedule.py```: Noise schedules
3. ```diffusion.py```: Forward/reverse diffusion
4. ```dataloader.py```: Dataloaders
5. ```utils.py```: LR scheduler, logging, `fsspec` handling
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748), AR transformer, and [Mamba](https://arxiv.org/abs/2312.00752)
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules
8. ```scripts/```: Shell scripts for training/evaluation


<a name="getting_started"></a>

## Getting started in this repository

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f requirements.yaml
conda activate mdlm
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```
and run the training as a batch job:
```bash
sbatch scripts/train_owt_mdlm.sh
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
* `analytic`: Analytic sampler proposed in SEDD.

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


## Citation
```
@misc{sahoo2024simple,
      title={Simple and Effective Masked Diffusion Language Models}, 
      author={Subham Sekhar Sahoo and Marianne Arriola and Yair Schiff and Aaron Gokaslan and Edgar Marroquin and Justin T Chiu and Alexander Rush and Volodymyr Kuleshov},
      year={2024},
      eprint={2406.07524},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
