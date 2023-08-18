import sys, argparse

from src.dense_retriever import *
from src.model_fine_tuning_functions import *
from src.utils import *



def main(qrels_path: str, hard_negatives_path: str):

    train, val, test = split_qrels_train_test(qrels_path)
    hard_negatives = read_hard_negatives(hard_negatives_path)
    train_hn, val_hn, test_hn = split_hard_negatives_train_val_test(hard_negatives)


    concat_train = pd.concat([train, train_hn])
    concat_val = pd.concat([val, val_hn])
    concat_test = pd.concat([test, test_hn])


    concat_train.to_csv('../data/train.csv', index=False)
    concat_val.to_csv('../data/val.csv', index=False)
    concat_test.to_csv('../data/test.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-q',dest='qrels_path',action='store',required=False,type=str,default="../data/qrels.txt")
    parser.add_argument('-hn',dest='hard_negatives_path',action='store',required=False,type=str,default="../data/bm25_hard_negatives_all.json")

    args = parser.parse_args()

    main(args.qrels_path, args.hard_negatives_path)