import os
import numpy as np
import pdb
import json
import pandas as pd
import string



if not os.path.exists('mapping/id_to_uri_graph_phenomNet11.dict') or not os.path.exists('mapping/uri_to_word_graph_phenomNet11.dict'):
	# URI to Unique Word Mapping text
	with open('mapping/uri_text_mapping.dict','r') as f:
		uri_clean_text = json.load(f)
	uri_rev_map_text = {v:u for u,v in uri_clean_text.items()}

	# URI to ID in graph corpus
	mapping = pd.read_csv('graph_data/mapping_phenomNet11.txt', sep='\t', header=None)
	mapping = mapping.values
	id_ = mapping[:,1]
	uri_ = mapping[:,0]

	uri_id_graph = {i:ur for i,ur in zip(id_,uri_)}
	uri_graph_rev_map = {v:u for u,v in uri_id_graph.items()}

	# Map ID in corpus file to unique word, same in text 
	id_word_combined_corpus = dict()
	uri_word_combined_corpus = dict()
	for id_,uri in uri_id_graph.items():
		if uri in uri_clean_text:
			id_word_combined_corpus[id_] = uri_clean_text[uri]
			uri_word_combined_corpus[uri] = uri_clean_text[uri]
		else:
			repl_word = '_'.join(str(el) for el in uri.rsplit('/')[-2:])
			id_word_combined_corpus[id_] = repl_word
			uri_word_combined_corpus[uri] = repl_word
	id_word_combined_corpus[2147483647] = '2147483647'
	with open('mapping/id_to_uri_graph_phenomNet11.dict','w') as f:
		j = json.dumps(id_word_combined_corpus)
		f.write(j)
	with open('mapping/uri_to_word_graph_phenomNet11.dict','w') as f:
		j = json.dumps(uri_word_combined_corpus)
		f.write(j)

else:
	with open('mapping/id_to_uri_graph_phenomNet11.dict','r') as f:
		id_word_combined_corpus = json.load(f)
	with open('mapping/uri_to_word_graph_phenomNet11.dict','r') as f:
		uri_word_combined_corpus = json.load(f)

with open('corpus/preprocessed_corpus_phenomNet11.txt','w') as f1:
	for line in open('graph_data/walks_phenomNet11.txt','r'):
		line = ' '.join([id_word_combined_corpus[id_] for id_ in line.split()]).encode('utf-8')
		f1.write(line+'\n')
