#!/bin/bash
# ^ This file is a bash script: a list of terminal commands run one after another.
#----------------------------------------------------------------------
# Transfer-learning experiment on one GPU on UVA's HPC system (Rivanna/Afton).
#
# Edit ONE line before submitting: the allocation (-A). Find yours with `allocations`.
# Then, from the repository folder:
#     sbatch ml/rivanna_job.slurm
# Check progress with:  squeue -u $USER   and   tail -f transfer-<jobid>.out
#
# Uses UVA's PyTorch container, so no PyTorch installation is needed.
#
# Lines starting with "#SBATCH" are read by Slurm (the queue system) as the
# job's settings. Bash ignores them because they start with "#".
#----------------------------------------------------------------------

# Allocation (account) charged for the computer time. EDIT this.
#SBATCH -A YOUR_ALLOCATION
# Use the group of machines that have GPUs.
#SBATCH -p gpu
# Ask for one GPU.
#SBATCH --gres=gpu:1
# Ask for 4 CPU cores (they prepare data while the GPU trains).
#SBATCH -c 4
# Ask for 32 GB of memory (RAM).
#SBATCH --mem=32G
# Maximum run time: 4 hours (the job is stopped if it runs longer).
#SBATCH -t 04:00:00
# Job name shown in squeue.
#SBATCH -J impostor-transfer
# File for normal output (%A becomes the job number).
#SBATCH -o transfer-%A.out
# File for error messages.
#SBATCH -e transfer-%A.err
# Skip these two machines: their old V100 GPUs do not work with recent PyTorch.
#SBATCH -x udc-an33-[37-38]

# Switch off all software modules for a clean start.
module purge
# Switch on Apptainer (runs containers) and PyTorch.
module load apptainer pytorch
# Find the newest PyTorch container file (.sif) and store its name in SIF.
SIF=$(ls $CONTAINERDIR/pytorch-*.sif | sort -V | tail -1)
# Print the container name to the log.
echo "Using container $SIF"

# Go to the folder where you typed sbatch (the repository folder).
cd ${SLURM_SUBMIT_DIR}
# Check that matplotlib (for plots) exists in the container; if not (||), install it.
apptainer exec --nv $SIF python -c "import matplotlib" 2>/dev/null || \
    apptainer exec $SIF pip install --user --quiet matplotlib

# Download the pretrained ORACLE-1 weights (skipped if already downloaded).
bash ml/get_oracle_weights.sh

# Print which GPU the job got and its memory.
nvidia-smi

# Run the experiment inside the container on the GPU (--nv lets it use the GPU;
# -u prints progress to the log right away; "\" continues the command on the next line).
#   --counts       numbers of labeled SN impostors to train with
#   --n-other      supernovae and AGN per class in training
#   --n-test       test objects per class, never seen in training
#   --seeds        repeats with different random starts, for error bars
#   --steps        training steps per run
#   --batch-size   light curves processed at once
#   --device cuda  use the GPU
#   --out          folder where results are saved
apptainer exec --nv $SIF python -u ml/finetune_experiment.py \
    --counts 25 50 100 250 1000 4000 --n-other 4000 --n-test 1500 \
    --seeds 3 --steps 3000 --batch-size 256 --device cuda --out ml/results_transfer
