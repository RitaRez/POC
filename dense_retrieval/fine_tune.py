import sys, argparse

from dense_retriever import *
from model_fine_tuning_functions import *
from utils import *



def main(batch_size: int, epochs: int, max_length: int):

    corpus = read_corpus(max_length, '../data/documents.json')
    queries = read_queries(max_length, '../data/queries.json')


    train, val, test = split_qrels_train_test('../data/qrels.txt')

    hard_negatives = read_hard_negatives('../data/bm25_hard_negatives_all.json')
    train_hn, val_hn, test_hn = split_hard_negatives_train_val_test(hard_negatives)



    concat_train = pd.concat([train, train_hn])
    concat_val = pd.concat([val, val_hn])
    concat_test = pd.concat([test, test_hn])


    train_data = prepare_train_set(corpus, train, queries)
    sentences1, sentences2, scores = prepare_validation_set(corpus, val, queries)

    model = train_model(batch_size, epochs, train_data, '../models/model_finetuned', sentences1, sentences2, scores)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-b',dest='batch_size',action='store',required=False,type=int,default=64)
    parser.add_argument('-e',dest='epochs',action='store',required=False,type=int,default=25)
    parser.add_argument('-m',dest='max_length',action='store',required=False,type=int,default=512)

    args = parser.parse_args()

    main(args.batch_size, args.epochs, args.max_length)