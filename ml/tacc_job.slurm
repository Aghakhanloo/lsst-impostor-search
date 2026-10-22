#!/bin/bash
#----------------------------------------------------------------------
# Train the light-curve transformer on one TACC GPU node.
#
# Before submitting, edit the three lines marked EDIT:
#   - the partition (queue) for your system
#   - your allocation (project) name
#   - the path to the cloned repository
#
# Partitions on common TACC systems (check `sinfo -s` or the user guide):
#   Lonestar6: gpu-a100  (or gpu-a100-dev for runs under 2 h)
#   Frontera:  rtx       (or rtx-dev)
#   Vista:     gh        (or gh-dev)
#
# Submit with:  sbatch ml/tacc_job.slurm
# Check with:   squeue -u $USER
#----------------------------------------------------------------------
#SBATCH -J impostor-lc
#SBATCH -o impostor-lc.%j.out
#SBATCH -e impostor-lc.%j.err
#SBATCH -p gpu-a100            # EDIT: partition
#SBATCH -A YOUR_ALLOCATION     # EDIT: allocation / project name
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

cd $HOME/lsst-impostor-search  # EDIT: path to the repository

module load python3 2>/dev/null || true
python3 -m venv $WORK/venv-impostor 2>/dev/null || true
source $WORK/venv-impostor/bin/activate
pip install --quiet -r ml/requirements.txt

nvidia-smi
python3 ml/train.py --n-per-class 50000 --epochs 30 --ensemble 5 --batch-size 512 \
    --device cuda --out ml/results
