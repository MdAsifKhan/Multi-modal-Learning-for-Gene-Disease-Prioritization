#!/bin/bash

#SBATCH --job-name=Weight-pass-Word2Vec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=hohndor_group
#SBATCH --time=20-00:00:00
#SBATCH --array=0-9

values=$(grep "^${SLURM_ARRAY_TASK_ID}:" text-weight.txt)
param1=$(echo $values | cut -f 2 -d:)


input_text='../multimodel_learning/corpus/preprocessed_medline_medline-annotated.txt'
input_graph='../multimodel_learning/corpus/preprocessed_corpus_phenomNet11.txt'
output_text='../multimodel_learning/embeddings/graph-text-weight-pass/weight_pass_text_phenomNet11_'+$param1+'.txt'
output_graph='../multimodel_learning/embeddings/graph-text-weight-pass/weight_pass_graph_phenomNet11_'+$param1+'.txt'

python weight_pass.py --file_text $input_text --file_graph $input_graph --embedding_size 512 --window_size_text 5 --window_size_graph 10 --weight_passes 5 --sg 1 --text_weight $param1 --file_output_text $output_text --file_output_graph $output_graph

