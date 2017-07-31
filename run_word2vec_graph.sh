#!/bin/bash

#SBATCH --job-name=Word2Vec-Graph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --qos=hohndor_group


inputfile='../multimodel_learning/graph_data/walks_phenomNet2.txt'
outputfile='../multimodel_learning/embeddings/graph-embeddings/embeddings_phenomNet2.txt'


python word2vec_onlygraph.py --corpus_file $inputfile --output_file $outputfile --embedding_size 512 --window_size 10 --sg 1

