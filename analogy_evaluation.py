import pandas as pd
import pdb
from itertools import cycle, product, combinations
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import auc
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator


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


def most_similar(positive, negative):
    # Build a "mean" vector for the given positive and negative terms
    mean_vecs = []
    for word in positive: mean_vecs.append(disease_reprsn[word2id_disease[word]])
    for word in negative: mean_vecs.append(-1 * gene_reprsn[word2id_gene[word]])
    
    mean = np.array(mean_vecs).mean(axis=0)
    mean /= np.linalg.norm(mean)

    # Now calculate cosine distances between this mean vector and all others
    dists = np.dot(gene_reprsn, mean)   

    best = np.argsort(dists)[::-1]
    result = [id2word_gene[i] for i in best if (id2word_gene[i] not in negative)]
    
    return result


def create_query_triples(disease_gene_map, k):
        negative_1 = list()
        negative_target = list()
        monogenic = [disease for disease in disease_gene_map if len(disease_gene_map[disease])==1]

        positive = set()
        for disease in monogenic:
		sim_vec = disease_sim_matrix[disease_common.index(disease)]
                sort_sim = np.argsort(sim_vec)[::-1]

		query_disease = list()
		for arg in sort_sim:
			if (disease_common[arg] in monogenic and disease_gene_map[disease][0]!= disease_gene_map[disease_common[arg]][0]):
                		query_disease.append(disease_common[arg])
			if len(query_disease) == k:
				break
		positive |= {pair for pair in product([disease], query_disease)}
		
        print('Number of Monogenic disease',len(monogenic))
        positive = [list(pair) for pair in list(positive)]

        for pair in positive:
                disease1 = pair[0]
                disease2 = pair[1]
                #gene1 = random.choice(disease_gene_map[disease1])
           
		negative_1.append(disease_gene_map[disease1][0])
                negative_target.append(disease_gene_map[disease2])
        return (positive, negative_1, negative_target)



EMBEDDING_ROOT_GRAPH= '../../../Documents/multimodel_data/data/'

map_graph = pd.read_csv(EMBEDDING_ROOT_GRAPH + 'mapping/mapping_hpo1.txt', sep='\t',header=None)
uri_ = map_graph[0].values
id_ = map_graph[1].values
id_uri = {i:ur for i,ur in zip(id_,uri_)}


embed_gene = pd.read_csv(EMBEDDING_ROOT_GRAPH + 'embeddings/graph_only/gene_embedding_hpo1.txt',header=None, sep='\t')
embed_gene_id =[gene.rsplit('/')[-1] for gene in embed_gene[0].values]
embed_gene = {gene:vec for gene,vec in zip(embed_gene_id, embed_gene.drop(embed_gene.columns[[0]],axis=1).values)}


embed_do = pd.read_csv(EMBEDDING_ROOT_GRAPH + 'embeddings/graph_only/diseases_embedding_hpo1.txt',header=None, sep='\t')
embed_do_id = [d_id.rsplit('/')[-1] for d_id in embed_do[0].values]
embed_do = {d_id.rsplit('/')[-1] : np.array(vec, dtype='float32') for d_id, vec in zip(embed_do_id,embed_do.drop(embed_do.columns[[0]],axis=1).values)}

genes_text = np.genfromtxt(EMBEDDING_ROOT +'common_genes_graph_text.txt', dtype = 'str')

gene_reprsn = list()
gene_set = list()
for gene in genes_text:
	if gene in embed_gene:
            gene_reprsn.append(embed_gene[gene])
            gene_set.append(gene)

disease_reprsn = list()
disease_common = list()

with open(EMBEDDING_ROOT + 'monogenic_diseases.dict', 'r') as f:
	monogenic_diseases = json.load(f)

for disease in monogenic_diseases:
        if disease in embed_do:
                disease_reprsn.append(embed_do[disease])
                disease_common.append(disease)


gene_reprsn = np.array(gene_reprsn, dtype='float32')
disease_reprsn = np.array(disease_reprsn, dtype='float32')
disease_sim_matrix = cosine_similarity(disease_reprsn)
np.fill_diagonal(disease_sim_matrix, 0)

#Create all triples Di,Dj,Gi
# k: Top Similar Dj
k = 50
positive, negative_1, negative_target = create_query_triples(disease_gene_map, gene_disease_map, k)

word2id_gene = {v:i for i,v in enumerate(gene_common)}
id2word_gene = {n:v for v,n in word2id_gene.items()}
word2id_disease = {v:i for i,v in enumerate(disease_common)}
id2word_disease = {n:v for v,n in word2id_disease.items()}


label_mat = dict()
TP_ = dict()
FP_ = dict()

for pos, neg, neg_t in zip(positive, negative_1, negative_target):	
	results = most_similar(pos, [neg])
	label_vec = [0]*len(results)
	for el1 in neg_t:
		idx = results.index(el1)
		label_vec[idx] = 1
	key = ''.join(el for el in pos) + neg[0]
	label_mat[key] = label_vec
	TP_[key] = np.cumsum(label_vec)
        FP_[key] = negcum(label_vec)			

tp_arr = TP_.values()
fp_arr = FP_.values()
tp_arr = np.array(tp_arr, dtype = "float32")
fp_arr = np.array(fp_arr, dtype = "float32")
tpsum = np.sum(tp_arr, axis = 0)
fpsum = np.sum(fp_arr, axis = 0)
tpr_ = tpsum/max(tpsum)
fpr_ = fpsum/max(fpsum)
avg_auc_ = auc(fpr_, tpr_)

print('Number of Analogy Task',len(label_mat))
print('Average auc scores across all analogies: ',avg_auc_)
np.savetxt(EMBEDDING_ROOT+'evaluation_results/OMIM_MGI1/graph_only_auc_pheno1.txt', auc_data , fmt = '%s')
pdb.set_trace()
