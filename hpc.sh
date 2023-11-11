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
PP=5
#QB=-1
#python train_scalar.py --id -1 --pp $PP --ep 100 --qbit $QB
#for i in {0..15}
#do
#  echo "$i"
#  python train_scalar.py --id "$i" --pp $PP --resume --ep 100 --qbit $QB
#done
##python train_100.py --id 0 --resume --ep 100
#python test_scalar.py --pp $PP --qbit $QB

python train_100.py --id -1 --skip_quant

#NP=1
#EB=2048
#python train_100.py --id -1 --pp $PP --ep 100 --n_parts $NP --n_embed $EB
#for i in {0..15}
#do
#  echo "$i"
#  python train_scalar.py --id "$i" --pp $PP --resume --ep 100 --n_parts $NP --n_embed $EB
#done
##python train_100.py --id 0 --resume --ep 100
#python test_scalar.py --pp $PP --n_parts $NP --n_embed $EB