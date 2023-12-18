#!/bin/bash
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=mp
#SBATCH --output=%x.out
#SBATCH --mail-type=END
#SBATCH --mail-user=dy2242@nyu.edu

module purge

singularity exec --nv \
            --overlay /scratch/dy2242/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
            python PIPE_main.py
            "