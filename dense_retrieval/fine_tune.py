import sys, argparse

from src.dense_retriever import *
from src.model_fine_tuning_functions import *
from src.utils import *



def main(batch_size: int, epochs: int, max_length: int):

    corpus = read_corpus(max_length, '../data/documents.json')
    queries = read_queries(max_length, '../data/queries.json')


    train = pd.read_csv('../data/train.csv')
    val = pd.read_csv('../data/val.csv')


    train_data = prepare_train_set(corpus, train, queries)
    sentences1, sentences2, scores = prepare_validation_set(corpus, val, queries)

    model = train_model(batch_size, epochs, train_data, '../models/model_finetuned', sentences1, sentences2, scores)


    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-b',dest='batch_size',action='store',required=False,type=int,default=64)
    parser.add_argument('-e',dest='epochs',action='store',required=False,type=int,default=25)
    parser.add_argument('-l',dest='max_length',action='store',required=False,type=int,default=512)

    args = parser.parse_args()

    main(args.batch_size, args.epochs, args.max_length)