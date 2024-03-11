#!/bin/bash --login
#$ -cwd
#SBATCH --job-name=CME_Detection
#SBATCH --out=base_model.out.%J
#SBATCH --err=base_model.err.%J
#SBATCH -p gpu
#SBATCH --gres=gpu:1

for ret in 0 1 2; do
	for fod in 0 1 2 3 4; do
		for model in Resnet121 Densenet121; do
			python CrossView.py \
				--backbone $model\
	        		--repeat $ret\
				--test_fold $fod \
                                --mode inference

		done
	done	
done
