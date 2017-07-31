import re
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import string
import multiprocessing
from nltk.corpus import stopwords
import sys
import os
import pdb
import json


cachedStopWords = stopwords.words('english')
#tokenizer = RegexpTokenizer(r,'\w+')
DATA_ROOT = 'text/'
def preprocess_text(filename):
	mapping = dict()
	with open(DATA_ROOT + 'preprocessed_medline_' + filename,'w') as f1:
		for line in open(os.path.join(DATA_ROOT, filename)):
			url  = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
			for uri in url:
				uri_split = uri.rsplit('/')
				id_ = uri_split[-1].split(';')
				uri_ = '/'.join(el for el in uri_split[:-1])
				repl_string = ' '
				for i in id_:
					new_uri = uri_ + '/' + i
					repl_ = '_'.join(str(el) for el in new_uri.rsplit('/')[-2:])
					repl_string = repl_ + ' ' + repl_string
					mapping[new_uri] = repl_
				line = line.replace(uri, repl_string)
			line = ' '.join([word for word in line.split() if word not in cachedStopWords])
			line = line.lower()
			line = line.translate(None, string.punctuation)
			f1.write(line+'\n')
	with open(DATA_ROOT + 'uri_word_mapping.dict','w') as f:
		j = json.dumps(mapping)
		f.write(j)

if __name__ == '__main__':
	filename = 'medline-annotated.txt'
	preprocess_text(filename)

