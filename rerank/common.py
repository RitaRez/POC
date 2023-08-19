import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
from pyserini.search import FaissSearcher
import torch
from pygaggle.rerank.base import Query, Text
import numpy as np
import json


def get_bm25(test_queries: str = '../data/queries.json', index_filepath: str = 'indexes/corpus_index', k: int = 1000):
    """
    
    """


    test_queries = pd.read_json(test_queries, lines=True)
    output_df = pd.DataFrame(columns=["QueryId",'EntityId'])

    searcher = LuceneSearcher(index_filepath)

    output_df = get_hits(searcher,test_queries).sort_values(by=['QueryId','Score'],ascending=[True,False])[['QueryId','EntityId', 'Score']]

    return output_df


def format_trec_style(df_to_format: str, submission_name: str) -> str:
    """
    Receives an csv containing data as query, document, score for each line and returns in format:
    1 Q0 pid1    1 2.73 runid1
    where we use white space to separate columns and the columns are:
    query_id Q0 document_id rank score run_id
    we will return the filepath in which the file was saved
    """

    df = pd.read_csv(df_to_format, sep=',', header=None)

    df['rank'] = df.groupby('QueryId')['Score'].rank(method='first', ascending=False)
    df['run_id'] = 'run_id'
    df['Q0'] = 'Q0'
    df['pid'] = df['EntityId']
    df['score'] = df['Score']
    df['query_id'] = df['QueryId']
    df = df[['query_id', 'Q0', 'pid', 'rank', 'score', 'run_id']]

    filepath = f'data/{submission_name}.txt'

    df.to_csv(filepath, sep=' ', header=False, index=False)
    
    return filepath


def get_hits(searcher, queries: pd.DataFrame, k: int = 100):
    """
    Receives the index and queries and returns the bm25 bests for each query
    """
    
    hits = searcher.batch_search(queries['description'].tolist(),queries['id'].tolist(),k=k,threads=10)
    
    scores={}
    for query in queries['id'].tolist():
        scores[query] = {}
        for hit in hits[query]:
            scores[query][hit.docid] = [hit.score]
    
    features = pd.DataFrame({(i,j): scores[i][j] 
                             for i in scores.keys() 
                             for j in scores[i].keys()}).T.reset_index()
    features.columns = ["QueryId","EntityId",'Score']
    return features


def parse_data(documents_file: str = '../data/documents.json'):
    """
    
    """
    for corpus in pd.read_json(documents_file,lines=True,encoding='utf-8',dtype={"id": str},chunksize=10**5):
        corpus['contents'] = corpus['title'] + ' ' + corpus['text'] #+ ' ' + corpus['keywords']
        with open('../data/official_data/documents.jsonl','a',encoding='utf-8') as file:
            for line in corpus[['id','contents']].to_dict(orient='records'):
                json.dump(line,file,ensure_ascii=False)
                file.write('\n')
    del corpus


def rerank(reranker, output, queries_filepath: str = '../data/queries.json') -> list:
    """
    
    """


    test_queries, doc_index = get_data()

    test_queries = pd.read_json(queries_filepath, lines=True)
    test_queries.head()
    
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

    return rankings


def get_data(searcher, test_queries):
    """ 
    """

    doc_index = {}
    output = get_hits(searcher,test_queries,k=1000)


    needed=set(output.sort_values(by='EntityId')['EntityId'].tolist())
    with open('corpus.jsonl',encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            if doc['id'] in needed:
                doc_index[doc['id']] = doc['text']
    del needed

    return test_queries, doc_index