import json, torch, time
import pandas as pd

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample


def read_corpus(corpus_path: str) -> dict[str]:
    """
    Read the corpus from file_path
    """
    
    films_data = {}
    with open(corpus_path, 'r') as f:
        for line in f:    
            film = json.loads(line)            
            # films_data[film['id']] =  film['text'] # film['title'] + ' ' + film['text'] + ' ' + film['genres'] + ' ' + film['startYear']
            new_text = film['text'].split()[:80]
            films_data[film['id']] = ' '.join(new_text)

    return films_data


def read_queries(queries_path: str) -> dict[str]:
    """
    Read the queries from file_path
    """

    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:    
            query = json.loads(line)            
            queries[query['id']] = query['title'] + ' ' + query['description']

    return queries


def read_hard_negatives(hard_negatives_path: str) -> pd.DataFrame:
    """
    Read the hard negatives from file_path
    """

    hard_negatives = {}
    with open(hard_negatives_path, 'r') as f:
        hard_negatives = json.load(f)  
                  

    rows = []
    for query, docs in hard_negatives.items():
        for doc in docs:
            rows.append({'query': query, 'doc': doc, 'label': 0})

    return pd.DataFrame(rows)


def split_qrels_train_test(qrels_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the corpus into train and test
    """
    
    df = pd.read_csv(qrels_path, sep="\s+", names=["query_id", "_", "movie_id", "label"]) # For header names
    df.drop('_', axis=1, inplace=True)

    train, val_test = train_test_split(df, test_size=0.4)
    val, test = train_test_split(val_test, test_size=0.5)

    return train, val, test


def split_hard_negatives_train_val_test(hard_negatives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the corpus into train and test
    """
    
    train, val_test = train_test_split(hard_negatives, test_size=0.4)
    val, test = train_test_split(val_test, test_size=0.5)

    return train, val, test


def prepare_train_set(corpus: dict, train: pd.DataFrame, queries: dict) -> list[InputExample]:
    """
    Prepare the train set
    """

    count = 0
    train_data = []; 
    
    for index, row in train.iterrows():  
        query_id = row['query_id']; movie_id = row['movie_id']; label = float(row['label'])
        if query_id not in queries or movie_id not in corpus:
            count += 1
            continue
        else:
            query = queries[query_id]
            film_text = corpus[movie_id]
            train_data.append(InputExample(texts=[query, film_text], label=label))
    

    print(f"Had to skip {count} pairs")
    
    return train_data


def prepare_validation_set(corpus: dict, val: pd.DataFrame, queries: dict) -> list[InputExample]:
    """
    For the validation while finetuning the model we need to return 3 pieces of information, two list with sentences and their respective score
    the first list will be of documents, the second of queries and the scores will be binary representing if they are relevant or not
    """

    count = 0
    sentences1 = []; sentences2 = []; scores = []
    
    for index, row in val.iterrows():  
        query_id = row['query_id']; movie_id = row['movie_id']; label = float(row['label'])
        if query_id not in queries or movie_id not in corpus:
            count += 1
            continue
        else:
            query = queries[query_id]
            film_text = corpus[movie_id]
            sentences1.append(query)
            sentences2.append(film_text)
            scores.append(label)
    

    print(f"Had to skip {count} pairs")
    

    return sentences1, sentences2, scores