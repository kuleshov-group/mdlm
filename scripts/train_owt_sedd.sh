#!/bin/bash
#SBATCH -J train_sedd                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.
srun python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=openwebtext-split \
  wandb.name=sedd-owt \
  parameterization=sedd \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  sampling.predictor=analytic \
  time_conditioning=True