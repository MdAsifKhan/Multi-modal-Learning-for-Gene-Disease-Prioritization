import pdb
import numpy as np
from sklearn.metrics import auc
import json



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


data_ = '../../../Documents/multimodel_data/data/'
sim_scores = np.loadtxt(data_+'sim_scores.txt', dtype = 'float32')
diseases = np.genfromtxt(data_+'diseases.txt', dtype = 'str')
genes = np.genfromtxt(data_+'genes.txt', dtype = 'str')

with open(data_+'disease_genes_human_text_eval.dict','r') as f:
	disease_genes = json.load(f)

gene_set = np.genfromtxt(data_+'common_genes_graph_text.txt', dtype = 'str')



# convert sim scores into nxm where n are genes and m are diseases
sim_mat = sim_scores.reshape((len(genes),len(diseases)))
sim_mat = sim_mat.T 


sim_dict = dict()
for i,dis in enumerate(diseases):
	sim_dict[dis] = sim_mat[i]



label_mat = dict()
for dis in sim_dict:
	if dis in disease_genes:
		assoc_genes = disease_genes[dis]
		s1 = list(set(assoc_genes))
		s1 = filter(None, s1)
		s2 = set(gene_set)

		if set(s1).intersection(s2):
			phenomNet_sim = sim_dict[dis]
			sort_similarity_arg = np.argsort(phenomNet_sim)[::-1]
			sort_gene = [gene_set[arg] for arg in sort_similarity_arg]
			label_vec = [0]*len(sort_gene)
			for gene in s1:
				if gene in sort_gene:
					label_vec[sort_gene.index(gene)] = 1
			label_mat[dis] = label_vec
	


array_tp = np.zeros((len(label_mat), len(gene_set)),dtype='float32')
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
print('Number of Disease {}'.format(len(label_mat)))
print('auc all associations {}'.format(auc(fpr_r, tpr_r)))
np.savetxt(data_ + 'evaluation_results/OMIM_MGI1/phenomNet_auc.txt', auc_data2, fmt = "%s")

pdb.set_trace()