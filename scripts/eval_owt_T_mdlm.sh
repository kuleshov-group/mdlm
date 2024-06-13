#!/bin/bash
#SBATCH -J T_mdlm                     # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-mulan-v2-scalar-owt-not-tZycWP-small-param-subs_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/60-1000000.ckpt

export HYDRA_FULL_ERROR=1

for T in 0 1000; do
  echo "$T"
  srun python -u -m main \
    mode=ppl_eval \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    T="$T" \
    eval.checkpoint_path=$checkpoint_path \
    +wandb.offline=true
done