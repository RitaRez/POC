from dense_retriever import *
from model_fine_tuning import *
from utils import *

corpus = read_corpus('../data/documents.json')
queries = read_queries('../data/queries.json')


train, val, test = split_qrels_train_test('../data/qrels.txt')

hard_negatives = read_hard_negatives('../data/bm25_hard_negatives_all.json')
train_hn, val_hn, test_hn = split_hard_negatives_train_val_test(hard_negatives)



concat_train = pd.concat([train, train_hn])
concat_val = pd.concat([val, val_hn])
concat_test = pd.concat([test, test_hn])


train_data = prepare_train_set(corpus, train, queries)
sentences1, sentences2, scores = prepare_validation_set(corpus, val, queries)

model = train_model(train_data, '../models/model_finetuned', sentences1, sentences2, scores)

