import sys, argparse

from dense_retrieval.src.dense_retriever import *
from dense_retrieval.src.model_fine_tuning_functions import *
from dense_retrieval.src.dense_retriever import *
from dense_retrieval.src.utils import *



def main(corpus_path: str, index_path: str, embeddings_path: str, max_length: int = 512):

    index = load_index("../data/index.bin")

    test = pd.read_csv('../data/test.csv')

    ans = retrieve_from_index(index, test, corpus_path, embeddings_path, max_length, '../data/test_retrieved.csv')

    print(ans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c',dest='corpus_path',action='store',required=True,type=str)
    parser.add_argument('-m',dest='model_path',action='store',required=True,type=str)
    parser.add_argument('-e',dest='embeddings_path',action='store',required=True,type=str)
    parser.add_argument('-l',dest='max_length',action='store',required=False,type=int,default=512)

    args = parser.parse_args()

    main(args.corpus_path, args.model_path, args.embeddings_path, args.max_length)