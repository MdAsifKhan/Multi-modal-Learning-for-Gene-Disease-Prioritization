import pandas as pd
from itertools import cycle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
import h5py
import json
import pdb





DATA_ROOT = '../multimodel_learning/graph_data/'
EMBEDDING_ROOT_COUPLED = '../multimodel_learning/embeddings/graph-text-weight-pass/'

with open('../multimodel_learning/mapping/uri_to_word_graph_g2_negation.dict') as f:
        uri_word_graph = json.load(f)

word_graph_uri = {u:v for v,u in uri_word_graph.items()}


map_graph = pd.read_csv(DATA_ROOT + 'mapping_g2_negation.txt', sep='\t',header=None)
uri_ = map_graph[0].values
id_ = map_graph[1].values
id_uri = {i:ur for i,ur in zip(id_,uri_)}

omim_do = pd.read_csv(DATA_ROOT +'omim2do.txt', sep = '\t', header = None)
do = omim_do[0].values
omim = omim_do[1].values
omim2do = {m:d.replace(':','_') for m,d in zip(omim, do)}


if not os.path.exists(EMBEDDING_ROOT_COUPLED + 'gene_embedding_text_weightpass_0.9.txt') or not os.path.exists(EMBEDDING_ROOT_COUPLED + 'disease_embedding_text_weightpass_0.9.txt'):
        f1 = open(os.path.join(EMBEDDING_ROOT_COUPLED + 'gene_embedding_text_weightpass_0.9.txt'),'w')
        f2 = open(os.path.join(EMBEDDING_ROOT_COUPLED + 'disease_embedding_text_weightpass_0.9.txt'),'w')
        f = open(os.path.join(EMBEDDING_ROOT_COUPLED, 'weight_pass_text_embeddings_+0.9+_skipgram.txt'),'r')
        f.next()
        for line in f:
                values = line.split()
                word = values[0]
                embed = values[1:]
                embed = '\t'.join(str(el) for el in embed)
                if word=='2147483647':
                        continue
                else:
                        if word in word_graph_uri:
                                uri = word_graph_uri[word]
                        else:
                                continue
                        if 'http://www.ncbi.nlm.nih.gov/gene/' in uri:
                                f1.write(uri + '\t' + embed + '\n')
                        elif 'http://purl.obolibrary.org/obo/DOID_' in uri:
                                f2.write(uri + '\t' + embed + '\n')
                        else:
                                continue
        f1.close()
        f2.close()
        f.close()

