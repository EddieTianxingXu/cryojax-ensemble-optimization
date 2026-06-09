#!/bin/bash
#SBATCH --job-name="hsp90half1"
#SBATCH --output="logs/run_%a.out"
#SBATCH --error="logs/run_%a.err"
#SBATCH --partition gpuA100x4
#SBATCH --gpus=1
#SBATCH --mem=120000M
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --array=0%1
#SBATCH --account=bfly-delta-gpu

unset LD_LIBRARY_PATH
source /u/txu8/miniconda3/etc/profile.d/conda.sh

cd /u/txu8/project/cryojax-ensemble-optimization/ACCESS

conda activate cryojax_eo_env
export TF_GPU_ALLOCATOR=cuda_malloc_async 
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
export PYTHONPATH=$PYTHONPATH:/u/txu8/project/github/package-update/src

COMMANDS=(
    "run_ensemble_reweighting --config config.yaml --n_images_in_parallel 50000"
)


export PYTHONFAULTHANDLER=1

MAPPED_ID=$((SLURM_ARRAY_TASK_ID))

eval "${COMMANDS[$MAPPED_ID]}"

# for local use
# run_ensemble_reweighting --config config.yaml