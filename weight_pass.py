from gensim.models import Word2Vec
from random import shuffle
import pdb
from gensim.models.word2vec import LineSentence
import multiprocessing
import os
import sys
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

def trainWord2Vec(filename=None, buildvoc=1, passes=20, sg=1, size=100,
                  dm_mean=0, window=5, hs=1, negative=5, min_count=1, workers=4):
    
    model = Word2Vec(size=size, sg=sg, window=window,
                     hs=hs, negative=negative, min_count=min_count, workers=workers)

    if buildvoc == 1:
        print('Building Vocabulary')

        model.build_vocab(LineSentence(filename))  # build vocabulary 

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        #lines = open(filename).readlines()
        #shuffle(lines)
        #open(filename,'w').writelines(lines)
        model.train(LineSentence(filename))
    return model


class TriDNR:

    def __init__(self, file_text=None,file_graph=None, window_text=5, window_graph=10, textweight=0.8, size=300, seed=1,sg=0, workers=1, passes=10, dm=0, min_count=3):
        
        # Initialize Word2Vec 
        text = trainWord2Vec(file_text, sg=sg, buildvoc=1, passes=1, size=size, window=window_text,workers=workers)
        graph = trainWord2Vec(file_graph, sg=sg, buildvoc=1, passes=1, size=size, window=window_graph, workers=workers)

        self.text = text
        self.graph = graph

        self.train(text, graph, file_text, file_graph, passes=passes, weight=textweight, window_text=window_text, window_graph=window_graph)


    def setWeights(self, originalModel, destModel, model, weight=1):
        def parallel_update_graph(key, originalModel, destModel):
            #for key in keys_graph:
            #    if not keys_text.__contains__(key):
            #        continue
            index = originalModel.vocab[key].index # Word2Vec index, Text
            id = destModel.vocab[key].index # Word2Vec index, Graph
            destModel.syn0[id] = (1-weight) * destModel.syn0[id] + weight * originalModel.syn0[index]
            destModel.syn0_lockf[id] = originalModel.syn0_lockf[index]
        def parallel_update_text(key, originalModel, destModel):
            #for key in keys_graph:
            #    if not keys_text.__contains__(key):
            #        continue
            index = destModel.vocab[key].index # Word2Vec index, Text
            id = originalModel.vocab[key].index # Word2Vec index, Graph
            destModel.syn0[index] = (1-weight) * destModel.syn0[index] + weight * originalModel.syn0[id]
            destModel.syn0_lockf[index] = originalModel.syn0_lockf[id]

        if model=='text':
            print('Copy Weights from Text to Graph')
            keys_text = originalModel.vocab.keys()
            keys_graph = destModel.vocab.keys()
            #parallel_update_graph(keys_text,keys_graph,originalModel,destModel)
	    pr = [multiprocessing.Process(target=parallel_update_graph, args=(key, originalModel, destModel)) for key in keys_graph if keys_text.__contains__(key)]
	    for p in pr:
		p.start()
	    for p in pr:
		p.join()
        else: # orignialModel is of Graph
            print('Copy Weights from Graph to Text')
            keys_text = destModel.vocab.keys()
            keys_graph = originalModel.vocab.keys()
            #parallel_update_text(keys_text, keys_graph,originalModel,destModel)
            pr1 = [multiprocessing.Process(target=parallel_update_text, args=(key, originalModel, destModel)) for key in keys_graph if keys_text.__contains__(key)]
            for p in pr1:
                p.start()
            for p in pr1:
                p.join()                                                                                                                         

    def train(self, text, graph, file_text, file_graph, window_text=5, window_graph=10, passes=10, weight=0.9):

        for i in xrange(passes):
            print('Iterative Runing %d' % i)
            self.setWeights(text, graph, weight=weight, model='text')
            #Train Word2Vec

            #lines = open(file_graph).readlines()
            #shuffle(lines)
            #open(file_graph,'w').writelines(lines)
            print("Update W2V Graph...")
            graph.train(file_graph)
            self.setWeights(graph, text, weight=(1-weight),model='graph')

            print("Update W2V Text...")
            #lines = open(file_text).readlines()
            #shuffle(lines)
            #open(file_text,'w').writelines(lines)
            text.train(LineSentence(file_text))

def word2vec_(args):
        file_text = args.file_text
	file_graph = args.file_graph
        embedding = args.embedding_size
        window_text = args.window_size_text
	window_graph = args.window_size_graph
	weight_passes = args.weight_passes
	text_weight = args.text_weight
	file_output_text = args.file_output_text
	file_output_graph = args.file_output_graph
	sg = args.sg
	tridnr_model = TriDNR(file_text=file_text, file_graph=file_graph, window_text=window_text, window_graph=window_graph, sg=sg,size=embedding, textweight= text_weight, seed=random_state, workers=multiprocessing.cpu_count(), passes=weight_passes)

	text = tridnr_model.text
	graph = tridnr_model.graph
	text.save_word2vec_format(file_output_text, binary=False)
	graph.save_word2vec_format(file_output_graph, binary=False)


def main():
        parser = ArgumentParser('Weight-Pass-Word2vec', formatter_class = ArgumentDefaultsHelpFormatter,conflict_handler = 'resolve')
        parser.add_argument('--file_text', help='Text Corpus file')
	parser.add_argument('--file_graph', help='Graph Corpus file')
        parser.add_argument('--embedding_size', type=int, help='Embedding size for word2vec')
        parser.add_argument('--window_size_text', type=int, help='Window Size for text word2vec')
        parser.add_argument('--window_size_graph', type=int, help='Window Size for graph word2vec')
        parser.add_argument('--weight_passes', type=int, help='Number of passes')
	parser.add_argument('--text_weight', type=float, help='Weight for Weight Passing')
	parser.add_argument('--file_output_text', help='Output file for Text Embeddings')
	parser.add_argument('--file_output_graph', help='Output file for Graph Embeddings')
        parser.add_argument('--sg', help ='Training Algorithm 0: CBOW, 1:Skipgram')
	args = parser.parse_args()
        word2vec_(args)



if __name__ == '__main__':
	random_state = 1
	sys.exit(main())

