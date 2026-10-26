#!/bin/bash
#----------------------------------------------------------------------
# Transfer-learning experiment on one GPU on UVA's HPC system (Rivanna/Afton).
#
# Edit ONE line before submitting: the allocation (-A). Find yours with `allocations`.
# Then, from the repository folder:
#     sbatch ml/rivanna_job.slurm
# Check progress with:  squeue -u $USER   and   tail -f transfer-<jobid>.out
#
# Uses UVA's PyTorch container, so no PyTorch installation is needed. The first
# lines install matplotlib into ~/.local if the container lacks it.
#----------------------------------------------------------------------
#SBATCH -A YOUR_ALLOCATION        # EDIT: your allocation name
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J impostor-transfer
#SBATCH -o transfer-%A.out
#SBATCH -e transfer-%A.err
#SBATCH -x udc-an33-[37-38]       # recent PyTorch containers do not support these V100 nodes

module purge
module load apptainer pytorch
SIF=$(ls $CONTAINERDIR/pytorch-*.sif | sort -V | tail -1)
echo "Using container $SIF"

cd ${SLURM_SUBMIT_DIR}
apptainer exec --nv $SIF python -c "import matplotlib" 2>/dev/null || \
    apptainer exec $SIF pip install --user --quiet matplotlib

# fetch the public ORACLE-1 weights if not present
bash ml/get_oracle_weights.sh

nvidia-smi
apptainer exec --nv $SIF python -u ml/finetune_experiment.py \
    --counts 25 50 100 250 1000 4000 --n-other 4000 --n-test 1500 \
    --seeds 3 --steps 3000 --batch-size 256 --device cuda --out ml/results_transfer
