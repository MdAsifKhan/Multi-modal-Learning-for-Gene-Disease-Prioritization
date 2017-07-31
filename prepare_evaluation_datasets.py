import pdb
import json
import numpy as np

mouse2homo = {}
homo2human = {}

data_ = '../../../Documents/multimodel_data/data/'
#better mapping


mgi2entrez = {}
with open(data_+'HMD_HumanPhenotype.rpt') as f:
	for line in f:
		items = line.strip().split()
		homo_id = items[2]
		human_entrez = items[1]
		if len(items) == 5:
		    mgi_id = items[4]
		else:
			mgi_id = items[5]

		mgi2entrez[mgi_id] = human_entrez


disease_genes_mouse = {}
disease_genes_human = {}
disease_genes_combined = {}
with open(data_ + 'MGI_OMIM.rpt') as f:
	for line in f:
		items = line.strip().split('\t')
		omim = 'OMIM:' + items[0]
		gene = items[6]
		mgi_id = items[-1]
		species_id = items[4]

		if species_id == '10090':
			if mgi_id in mgi2entrez:
				entrez = mgi2entrez[mgi_id]
				if omim in disease_genes_mouse:
					disease_genes_mouse[omim].append(entrez)
				else:
					disease_genes_mouse[omim] = [entrez]

		if species_id == '9606':
			if omim in disease_genes_human:
				disease_genes_human[omim].append(gene)
			else:
				disease_genes_human[omim] = [gene]


for item in disease_genes_human.keys():
	disease_genes_combined[item] = disease_genes_human[item]

for disease in disease_genes_mouse:
	if disease not in disease_genes_human:
		disease_genes_combined[disease] = disease_genes_mouse[disease]

#get the common ones with text and evaluation file
disease_set = np.genfromtxt(data_ +'common_omims_graph_text_eval_file.txt', dtype = 'str')
disease_genes_human_text_eval = {}
disease_genes_mouse_text_eval= {}
disease_genes_combined_text_eval = {}

for dis in disease_set:
	if dis in disease_genes_human:
		genes = disease_genes_human[dis]
		disease_genes_human_text_eval[dis] = genes

for dis in disease_set:
	if dis in disease_genes_mouse:
		genes = disease_genes_mouse[dis]
		disease_genes_mouse_text_eval[dis] = genes

for dis in disease_set:
	if dis in disease_genes_combined:
		genes = disease_genes_combined[dis]
		disease_genes_combined_text_eval[dis] = genes

with open(data_ + 'disease_genes_human_text_eval.dict','w') as f:
	j = json.dumps(disease_genes_human_text_eval)
	f.write(j)


with open(data_ + 'disease_genes_mouse_text_eval.dict', 'w') as f:
	j = json.dumps(disease_genes_mouse_text_eval)
	f.write(j)

with open(data_ + 'disease_genes_combined_text_eval.dict', 'w') as f:
	j = json.dumps(disease_genes_combined_text_eval)
	f.write(j)
