#!/bin/bash
#SBATCH -J zinc_linker_genertaion
#SBATCH -o zinc_linker_genertaion-%j.log
#SBATCH -e zinc_linker_genertaion-%j.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
##SBATCH -w  gpu04# If you want to specify a computing node, you can write its name here and remove the first # 

echo -n "Start at " >> zinc_linker_genertaion_time.log
date +%T%t%F >> zinc_linker_genertaion_time.log

python generation_linker.py

echo -n "End at " >> zinc_linker_genertaion_time.log
date +%T%t%F >> zinc_linker_genertaion_time.log
