# %%
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
from pyserini.search import FaissSearcher
import torch
from pygaggle.rerank.base import Query, Text
import numpy as np
import json

# %%
test_queries = pd.read_csv('test_queries.csv',dtype={"QueryId":str})
test_queries.head()

# %%
train_queries_relevance = pd.read_csv('train_qrels.csv',dtype={"QueryId":str,"EntityId":str})
train_queries_relevance.head()

# %%
train_queries = pd.read_csv('train_queries.csv',dtype={"QueryId":str})
train_queries.head()

# %%
train = pd.merge(train_queries_relevance,train_queries,on='QueryId')
train.head()

# %%
def getHits(searcher,queries,k=100):
    hits = searcher.batch_search(queries['Query'].tolist(),queries['QueryId'].tolist(),k=k,threads=10)
    
    scores={}
    for query in queries['QueryId'].tolist():
        scores[query] = {}
        for hit in hits[query]:
            scores[query][hit.docid] = [hit.score]
    
    features = pd.DataFrame({(i,j): scores[i][j] 
                             for i in scores.keys() 
                             for j in scores[i].keys()}).T.reset_index()
    features.columns = ["QueryId","EntityId",'Score']
    return features

def getRating(reader,df):
    scores=[]
    for line in df.to_dict(orient='records'):
        scores.append(reader.compute_bm25_term_weight(line['EntityId'],line['Query']))
    return scores

def normalize(column):
    return (column - column.min()) / (column.max()-column.min()) 

# %% [markdown]
# # Reranking with bert

# %%
#refine with bert
from pygaggle.rerank.transformer import MonoBERT
reranker =  MonoBERT()

# %%
searcher = LuceneSearcher('indexes/corpus_index')
#searcher.set_rm3(10, 10, 0.5) #set rm3
output = getHits(searcher,test_queries,k=1000)

# %%
#get needed documents
doc_index={}
curr_doc=0
needed=set(output.sort_values(by='EntityId')['EntityId'].tolist())
with open('corpus.jsonl',encoding='utf-8') as file:
    for line in file:
        doc = json.loads(line)
        if doc['id'] in needed:
            doc_index[doc['id']] = doc['text']
del needed

# %%
#rerank
rankings=[]
for q in test_queries.to_dict(orient='records'):
    print(q['Query'])
    
    #format query and text
    query = Query(q['Query'])
    texts=[]
    for hit in output[output['QueryId']==q['QueryId']].to_dict(orient='records'):
        texts.append(Text(doc_index[hit['EntityId']],{'docid':hit['EntityId']},0))
        
    reranked = reranker.rerank(query, texts)
    for i,r in enumerate(reranked):
        if i >= 100: #take the first 100 only
            break
        rankings.append([q['QueryId'],r.metadata['docid']])

# %%
output = pd.DataFrame(rankings)
output.columns = ['QueryId','EntityId']
output.to_csv('submissions/monobert_bm25_1000.csv',index=False)

# %% [markdown]
# ## Refine with duoT5

# %%
from pygaggle.rerank.transformer import DuoT5
reranker =  DuoT5()

# %%
rankings=[]
for q in test_queries.to_dict(orient='records'):
    print(q['Query'])
    
    #format query and text
    query = Query(q['Query'])
    texts=[]
    for hit in output[output['QueryId']==q['QueryId']].to_dict(orient='records'):
        texts.append(Text(doc_index[hit['EntityId']],{'docid':hit['EntityId']},0))
        
    reranked = reranker.rerank(query, texts)
    for i,r in enumerate(reranked):
        rankings.append([q['QueryId'],r.metadata['docid']])
output = pd.DataFrame(rankings)
output.columns = ['QueryId','EntityId']
output.to_csv('submissions/duot5_monobert_bm25_1000.csv',index=False)

# %% [markdown]
# # Reranking with monoT5

# %%
#refine with t5
from pygaggle.rerank.transformer import MonoT5
reranker =  MonoT5()

# %%
searcher = LuceneSearcher('indexes/corpus_index')
#searcher.set_rm3(10, 10, 0.5) #set rm3
output = getHits(searcher,test_queries,k=1000)

# %%
#get needed documents
doc_index={}
curr_doc=0
needed=set(output.sort_values(by='EntityId')['EntityId'].tolist())
with open('corpus.jsonl',encoding='utf-8') as file:
    for line in file:
        doc = json.loads(line)
        if doc['id'] in needed:
            doc_index[doc['id']] = doc['text']
del needed

# %%
#rerank
rankings=[]
for q in test_queries.to_dict(orient='records'):
    print(q['Query'])
    query = Query(q['Query'])
    texts=[]
    for hit in output[output['QueryId']==q['QueryId']].to_dict(orient='records'):
        texts.append(Text(doc_index[hit['EntityId']],{'docid':hit['EntityId']},0))
    reranked = reranker.rerank(query, texts)
    for i,r in enumerate(reranked):
        if i >= 100: #take the first 100 only
            break
        rankings.append([q['QueryId'],r.metadata['docid']])

# %%
output = pd.DataFrame(rankings)
output.columns = ['QueryId','EntityId']
output.to_csv('submissions/monoT5_bm25_1000.csv',index=False)

# %% [markdown]
# # Using weighted indexes

# %% [markdown]
# ## Combining dense and sparse top 1000

# %%
searcher = FaissSearcher(
    'encoding/shardfull',
    'facebook/dpr-question_encoder-multiset-base'
)
hits = getHits(searcher,test_queries,k=1000)
hits.rename(columns={'Score':'dense'},inplace=True)
del searcher
searcher = LuceneSearcher('indexes/corpus_index')
bm25 = getHits(searcher,test_queries,k=1000)
bm25.rename(columns={'Score':'bm25'},inplace=True)

hits = pd.merge(bm25,hits,on=['QueryId','EntityId'],how='outer')
hits = hits.fillna(0.0)

# %%
#normalize for each query
alpha = 0.8
beta = round(1-alpha,1)
out = []
#get top 1000 for each query
for q,gb in hits.groupby('QueryId'):
    gb['dense'] = normalize(gb['dense'])
    gb['bm25'] = normalize(gb['bm25'])
    gb['score'] = alpha*gb['bm25'] + beta*gb['dense']
    gb = gb.sort_values(by='score',ascending=False).reset_index(drop=True)
    out.append(gb[:1000])
output = pd.concat(out)

# %%
#refine with bert
from pygaggle.rerank.transformer import MonoBERT
reranker =  MonoBERT()

# %%
#get needed documents
doc_index={}
curr_doc=0
needed=set(output.sort_values(by='EntityId')['EntityId'].tolist())
with open('corpus.jsonl',encoding='utf-8') as file:
    for line in file:
        doc = json.loads(line)
        if doc['id'] in needed:
            doc_index[doc['id']] = doc['text']
del needed

# %%
#rerank
rankings=[]
for q in test_queries.to_dict(orient='records'):
    print(q['Query'])
    
    #format query and text
    query = Query(q['Query'])
    texts=[]
    for hit in output[output['QueryId']==q['QueryId']].to_dict(orient='records'):
        texts.append(Text(doc_index[hit['EntityId']],{'docid':hit['EntityId']},0))
        
    reranked = reranker.rerank(query, texts)
    for i,r in enumerate(reranked):
        if i >= 100: #take only first 100
            break
        rankings.append([q['QueryId'],r.metadata['docid']])

# %%
output = pd.DataFrame(rankings)
output.columns = ['QueryId','EntityId']
output.to_csv(f'submissions/weighted_{int(alpha*10)}bm25_{int(beta*10)}dense_monobert.csv',index=False)

# %% [markdown]
# ## Refine with duoT5

# %%
from pygaggle.rerank.transformer import DuoT5
reranker =  DuoT5()

# %%
rankings=[]
for q in test_queries.to_dict(orient='records'):
    print(q['Query'])
    
    #format query and text
    query = Query(q['Query'])
    texts=[]
    for hit in output[output['QueryId']==q['QueryId']].to_dict(orient='records'):
        texts.append(Text(doc_index[hit['EntityId']],{'docid':hit['EntityId']},0))
        
    reranked = reranker.rerank(query, texts)
    for i,r in enumerate(reranked):
        rankings.append([q['QueryId'],r.metadata['docid']])
output = pd.DataFrame(rankings)
output.columns = ['QueryId','EntityId']
output.to_csv(f'submissions/duot5_weighted_{int(alpha*10)}bm25_{int(beta*10)}dense_monobert.csv',index=False)


