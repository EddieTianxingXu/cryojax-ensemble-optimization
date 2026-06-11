#!/bin/bash
#SBATCH --job-name="hsp90fin"
#SBATCH --output="logs/run_%a.out"
#SBATCH --error="logs/run_%a.err"
#SBATCH --partition gpuA100x4
#SBATCH --gpus=1
#SBATCH --mem=120000M
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --array=10%1
#SBATCH --account=bfly-delta-gpu

unset LD_LIBRARY_PATH
source /u/txu8/miniconda3/etc/profile.d/conda.sh

cd /u/txu8/project/cryojax-ensemble-optimization/ACCESS

conda activate cryojax_eo_env
export TF_GPU_ALLOCATOR=cuda_malloc_async 
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
export PYTHONPATH=$PYTHONPATH:/u/txu8/project/github/package-update/src

COMMANDS=(
    "python -m get_sim -sbatch 512 --expt_dir /work/hdd/bfly/txu8/project/HSP90-p23/half1trunc -star /work/hdd/bfly/txu8/project/HSP90-p23/star/starfile_half1_truncate.star -mrc /work/hdd/bfly/txu8/project/HSP90-p23/particles/"
)


export PYTHONFAULTHANDLER=1

MAPPED_ID=$((SLURM_ARRAY_TASK_ID-10))

eval "${COMMANDS[$MAPPED_ID]}"

#for local 

# python -m extract_expt -mrc /home/eddie/Documents/ -star /home/eddie/Documents/STAR/EMD-50421_particles_truncate.star --expt_dir ./mrc_images
# python -m extract_expt -mrc /home/eddie/Documents/ -star /home/eddie/Documents/STAR/EMD-50421_particles_truncate.star --expt_dir ./mrc_images

# python -m extract_expt -mrc /home/eddie/Documents/Hsp90-p23/particles -star /home/eddie/Documents/Hsp90-p23/star/starfile_half1_tiny.star --expt_dir ./half1Images