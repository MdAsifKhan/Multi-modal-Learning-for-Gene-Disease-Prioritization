#!/bin/bash

#SBATCH --job-name=Concatenate-Corpus-Word2Vec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=hohndor_group
#SBATCH --time=3-00:00:00



inputdir='../multimodel_learning/corpus/concat_corpus_phenomNet22.txt'
outputfile='../multimodel_learning/embeddings/graph-text-concat-corpus/concat_corpus_embeddings_phenomNet22.txt'


python graph_concat_text_corpus.py --corpus_file $inputdir --output_file $outputfile --embedding_size 512 --window_size 10 --sg 1
