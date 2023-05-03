import json, torch, time, torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader



def read_corpus(corpus_path: str) -> dict[str]:
    """
    Read the corpus from file_path
    """
    
    films_data = {}
    with open(corpus_path, 'r') as f:
        for line in f:    
            film = json.loads(line)            
            films_data[film['id']] = film['title'] + ' ' + film['text'] #+ ' ' + film['genres'] + ' ' + film['startYear']

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


def split_qrels_train_test(qrels_path: str) -> tuple[pd.Dataframe, pd.Dataframe]:
    """
    Split the corpus into train and test
    """
    
    df = pd.read_csv(qrels_path, sep="\s+", names=["query_id", "_", "movie_id", "label"]) # For header names
    df.drop('_', axis=1, inplace=True)

    train, test = train_test_split(df, test_size=0.2)

    return train, test


def prepare_train_set(corpus: dict, queries: dict, qrels_path: str) -> list[InputExample]:
    """
    Prepare the train set
    """

    train, _  = split_qrels_train_test(qrels_path)

    train_data = []
    for index, row in train.iterrows():  
        query_id = row['query_id']; movie_id = row['movie_id']; label = row['label']
        train_data.append(InputExample(texts=[queries[query_id], corpus[movie_id]], label=label))

    return train_data


def train_model(corpus: dict, queries: dict, model_path: str):
    """
    Train the model
    """

    model = SentenceTransformer('bert-base-uncased')

    train_examples = prepare_train_set(corpus, queries)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    model.save(model_path)

    return model


def encode_corpus(corpus: dict, embedings_path: str, model_path: str):
    """
    Encode the corpus using SentenceTransformer
    """

    model = SentenceTransformer(model_path)
    embeddings = model.encode(corpus[:10])

    with open(embedings_path, "wb") as fOut:
        pickle.dump({'sentences': corpus, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    return embeddings


if __name__ == "__main__":

    if not os.isfile('embeddings.pkl'):
        
        corpus = read_corpus("../Movies/documents.json")
        queries = read_queries("../Movies/queries.json")
        
        if not os.isfile('../Models'):            
            train_model(corpus, queries)
        
        else:    
            embeddings = encode_corpus(corpus) 

    else:

        with open('embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_sentences = stored_data['sentences']
            embeddings = stored_data['embeddings']









