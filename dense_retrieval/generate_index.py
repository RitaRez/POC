import sys, argparse

from src.dense_retriever import *
from src.model_fine_tuning_functions import *
from src.dense_retriever import *
from src.utils import *



def main(corpus_path: str, model_path: str, embeddings_path: str, max_length: int = 512):

    corpus = read_corpus(max_length, corpus_path)
    print("Generating embeddings")
    embeddings = encode_corpus(corpus, model_path, embeddings_path)
    print("Building index")
    index = build_index(embeddings)

    index.save_index("../data/index.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c',dest='corpus_path',action='store',required=True,type=str)
    parser.add_argument('-m',dest='model_path',action='store',required=True,type=str)
    parser.add_argument('-e',dest='embeddings_path',action='store',required=True,type=str)
    parser.add_argument('-l',dest='max_length',action='store',required=False,type=int,default=512)

    args = parser.parse_args()

    main(args.corpus_path, args.model_path, args.embeddings_path, args.max_length)