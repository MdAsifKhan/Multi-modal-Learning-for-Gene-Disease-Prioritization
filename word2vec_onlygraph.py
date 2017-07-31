from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import multiprocessing
import sys
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

def word2vec_(args):
        input_file = args.corpus_file
        output_file = args.output_file
        embedding = args.embedding_size
	window = args.window_size
	sg = args.sg
        
        model = Word2Vec(LineSentence(input_file), size=embedding, sg=sg, window=window, min_count=1, workers=multiprocessing.cpu_count())
        model.save_word2vec_format(output_file, binary=False)


def main():
        parser = ArgumentParser('Word2vec-Graph', formatter_class = ArgumentDefaultsHelpFormatter,conflict_handler = 'resolve')
        parser.add_argument('--corpus_file', help='File with Graph Corpus')
        parser.add_argument('--output_file', help='file-name for saving embeddings')
        parser.add_argument('--embedding_size', type=int, help='Embedding size for word2vec')
        parser.add_argument('--window_size', type=int, help='Window Size for word2vec')
        parser.add_argument('--sg', type=int, help = 'Training Algorithm 0:CBOW 1:Skipgram')
	args = parser.parse_args()
        word2vec_(args)

if __name__ == '__main__':

    sys.exit(main())

