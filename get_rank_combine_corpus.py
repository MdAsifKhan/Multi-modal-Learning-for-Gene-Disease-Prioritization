import pandas as pd
import pdb
from itertools import cycle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import auc
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))


def negcum(rank_vec):
	rank_vec_cum = []
	prev = 0
	for x in rank_vec:
		if x == 0:
			x = x+1
			prev = prev + x
			rank_vec_cum.append(prev)
		else:
			rank_vec_cum.append(prev)
	rank_vec_cum = np.array(rank_vec_cum)
	return rank_vec_cum


EMBEDDING_ROOT = '../../../Documents/multimodel_data/data/'

embed_gene = pd.read_csv(EMBEDDING_ROOT + 'embeddings/text_graph/genes_concat_corpus_embeddings_pheno1.txt',header=None, sep=' ')
embed_gene_id =[gene[4:] for gene in embed_gene[0].values]
embed_gene = {g_id : np.array(vec, dtype='float32') for g_id, vec in zip(embed_gene_id,embed_gene.drop(embed_gene.columns[[0]],axis=1).values)}


embed_omim = pd.read_csv(EMBEDDING_ROOT + 'embeddings/text_graph/omim_concat_corpus_embeddings_pheno1.txt',header=None, sep=' ')
embed_omim_id = [d_id.split()[0] for d_id in embed_omim[0].values]
embed_omim = {d_id : np.array(vec, dtype='float32') for d_id, vec in zip(embed_omim_id,embed_omim.drop(embed_omim.columns[[0]],axis=1).values)}


with open(EMBEDDING_ROOT + 'disease_genes_mouse.dict','r') as f:
    disease_gene_mouse = json.load(f)

with open(EMBEDDING_ROOT + 'disease_genes_human.dict', 'r') as f:
	disease_gene_human = json.load(f)

with open(EMBEDDING_ROOT + 'disease_genes_combined.dict', 'r') as f:
	disease_gene_combined = json.load(f)


with open(EMBEDDING_ROOT +'disease_genes_human_text_eval.dict','r') as f:
	disease_genes = json.load(f)



#genes in both text and graph
genes_text = np.genfromtxt(EMBEDDING_ROOT +'common_genes_graph_text.txt', dtype = 'str')

gene_reprsn = list()
gene_set = list()
for gene in genes_text:
	if gene in embed_gene:
            gene_reprsn.append(embed_gene[gene])
            gene_set.append(gene)

gene_reprsn = np.array(gene_reprsn, dtype='float32')

label_mat = dict()
disease_set = disease_genes.keys()

for analyze_disease in disease_set:
	analyze_disease_gene_assocn = disease_genes[analyze_disease]
        s1 = list(set(analyze_disease_gene_assocn))
        s1 = filter(None, s1)
        s2 = set(gene_set)

	if set(s1).intersection(s2):
		disease_embed = list()
		if analyze_disease in embed_omim:
			disease_embed.append(embed_omim[analyze_disease])
			disease_embed = np.array(disease_embed, dtype='float32')
			similarity_gene_disease = cosine_similarity(gene_reprsn,disease_embed)
			similarity_gene_disease = similarity_gene_disease.flatten()
			prob_gene_disease  = sigmoid(similarity_gene_disease)

			sort_similarity_arg = np.argsort(prob_gene_disease)[::-1]
			sort_gene = [gene_set[arg] for arg in sort_similarity_arg]
			label_vec = [0]*len(sort_gene)
			for gene in s1:
				if gene in sort_gene:
					label_vec[sort_gene.index(gene)] = 1
			label_mat[analyze_disease] = label_vec


array_tp = np.zeros((len(label_mat),len(gene_set)),dtype='float32')
array_fp = np.zeros((len(label_mat), len(gene_set)), dtype = 'float32')

for i,row in enumerate(label_mat.values()):
	elem = np.asarray(row, dtype='float32')
	tpcum = np.cumsum(elem)
	fpcum = negcum(elem)  	
	array_tp[i] = tpcum
	array_fp[i] = fpcum	


#compute fpr and tpr Rob's way 
tpsum = np.sum(array_tp, axis = 0)
fpsum = np.sum(array_fp, axis = 0)
tpr_r = tpsum/max(tpsum)
fpr_r = fpsum/max(fpsum)
auc_data2 = np.c_[fpr_r, tpr_r]
print('Number of Disease {} '.format(len(label_mat)))
print('auc all associations {}'.format(auc(fpr_r, tpr_r)))
np.savetxt(EMBEDDING_ROOT + 'evaluation_results/OMIM_MGI1/pheno1_graph_text_concat_auc2.txt', auc_data2, fmt = "%s")

pdb.set_trace()

