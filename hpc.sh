#!/bin/bash
#
#SBATCH --job-name=ensemble
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#
#SBATCH --mail-type=END
#SBATCH --mail-user=cc6858@nyu.edu

source ~/.bashrc
cd /scratch/gilbreth/"$USER"/AdaptiveEnsemble || exit
conda activate d2l
#python train_100.py --id -1 --ep 100 --n_parts 2
for i in {0..15}
do
  echo "$i"
  python train_100.py --id "$i" --resume --ep 100 --n_parts 2
done
#python train_100.py --id 0 --resume --ep 100
python test_100.py --n_parts 2