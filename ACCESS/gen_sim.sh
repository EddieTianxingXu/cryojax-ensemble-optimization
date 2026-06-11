#!/bin/bash
#SBATCH --job-name="hsp90fin"
#SBATCH --output="logs/run_%a.out"
#SBATCH --error="logs/run_%a.err"
#SBATCH --partition gpuA100x4
#SBATCH --gpus=1
#SBATCH --mem=120000M
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --array=0-5%6
#SBATCH --account=bfly-delta-gpu

unset LD_LIBRARY_PATH
source /u/txu8/miniconda3/etc/profile.d/conda.sh

cd /u/txu8/project/cryojax-ensemble-optimization/ACCESS

conda activate cryojax_eo_env
export TF_GPU_ALLOCATOR=cuda_malloc_async 
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
export PYTHONPATH=$PYTHONPATH:/u/txu8/project/github/package-update/src

COMMANDS=(
    "python -m get_sim -sbatch 512 --sim_dir /work/hdd/bfly/txu8/project/HSP90-p23/finImagesUniform200 -star /work/hdd/bfly/txu8/project/HSP90-p23/star/starfile_half1_truncate.star -pdb /work/hdd/bfly/txu8/project/HSP90-p23/pdb/final_walker_0.pdb --rotmode uniform -rot 200"
    "python -m get_sim -sbatch 512 --sim_dir /work/hdd/bfly/txu8/project/HSP90-p23/finImagesGaussian200 -star /work/hdd/bfly/txu8/project/HSP90-p23/star/starfile_half1_truncate.star -pdb /work/hdd/bfly/txu8/project/HSP90-p23/pdb/final_walker_0.pdb --rotmode gaussian -rot 200"


    "python -m get_sim -sbatch 512 --sim_dir /work/hdd/bfly/txu8/project/HSP90-p23/initImagesUniform200 -star /work/hdd/bfly/txu8/project/HSP90-p23/star/starfile_half1_truncate.star -pdb /work/hdd/bfly/txu8/project/HSP90-p23/pdb/initial_model.pdb --rotmode uniform -rot 200"
    "python -m get_sim -sbatch 512 --sim_dir /work/hdd/bfly/txu8/project/HSP90-p23/initImagesGaussian200 -star /work/hdd/bfly/txu8/project/HSP90-p23/star/starfile_half1_truncate.star -pdb /work/hdd/bfly/txu8/project/HSP90-p23/pdb/initial_model.pdb --rotmode gaussian -rot 200"

)


export PYTHONFAULTHANDLER=1

MAPPED_ID=$((SLURM_ARRAY_TASK_ID))

eval "${COMMANDS[$MAPPED_ID]}"


# for local
# python -m get_sim -sbatch 8 --sim_dir /home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/PDB7DCXimages -star /home/eddie/Documents/STAR/EMD-50422_particles.star -pdb /home/eddie/Documents/PDB/50422_7DCX.pdb
# python -m get_sim -sbatch 8 --sim_dir /home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/PDB7YCZimages -star /home/eddie/Documents/STAR/EMD-50421_particles.star -pdb /home/eddie/Documents/PDB/50421_7YCZ.pdb

# python -m get_sim -sbatch 8 --sim_dir /home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/Hsp90finImages -pdb /home/eddie/Documents/Hsp90-p23/pdb/final_walker_0.pdb -star /home/eddie/Documents/Hsp90-p23/star/starfile_half1.star

# python -m get_sim -sbatch 8 --sim_dir /home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/Hsp90initImages -pdb /home/eddie/Documents/Hsp90-p23/pdb/initial_model.pdb -star /home/eddie/Documents/Hsp90-p23/star/starfile_half1.star