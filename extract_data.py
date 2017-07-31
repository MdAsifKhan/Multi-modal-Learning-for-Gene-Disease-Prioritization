import pdb
import os
import pandas as pd


EMBEDDING_ROOT_GRAPH = '../../../Documents/multimodel_data/data/'
map_graph = pd.read_csv(EMBEDDING_ROOT_GRAPH + 'mapping/mapping_phenomNet1.txt', sep='\t',header=None)
uri_ = map_graph[0].values
id_ = map_graph[1].values
id_uri = {i:ur for i,ur in zip(id_,uri_)}

if not os.path.exists(EMBEDDING_ROOT_GRAPH + 'graph_only/gene_embedding_phenomNet1.txt') or not os.path.exists(EMBEDDING_ROOT_GRAPH + 'graph_only/diseases_embedding_phenomNet1.txt'):

	f1 = open(os.path.join(EMBEDDING_ROOT_GRAPH + 'embeddings/graph_only/gene_embedding_phenomNet1.txt'),'w')
	f2 = open(os.path.join(EMBEDDING_ROOT_GRAPH + 'embeddings/graph_only/diseases_embedding_phenomNet1.txt'),'w')
	f = open(os.path.join( EMBEDDING_ROOT_GRAPH + 'embeddings/embeddings_phenomNet1.txt'),'r')
	f.next()
	for line in f:
		values = line.split()
		word = int(values[0])
		embed = values[1:]
		embed = '\t'.join(str(el) for el in embed)
		if word==2147483647:
			continue
		else:
			if 'http://www.ncbi.nlm.nih.gov/gene/' in id_uri[word]:
				f1.write(id_uri[word] + '\t' + embed + '\n')
			elif 'http://bio2vec.net/disease/OMIM:' in id_uri[word]:
				f2.write(id_uri[word] + '\t' + embed + '\n')
			else:
				continue
	f1.close()
	f2.close()
	f.close()