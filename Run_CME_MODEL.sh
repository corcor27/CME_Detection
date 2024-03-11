#!/bin/bash 

for ret in 0 1 2; do
	for fod in 0 1 2 3 4; do
		for model in Resnet121 Densenet121 ; do
			python CrossView.py \
				--backbone $model\
	        		--repeat $ret\
				--test_fold $fod \
                                --mode train

		done
	done	
done
