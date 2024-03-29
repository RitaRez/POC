import json, torch, time, math, pickle
import pandas as pd

from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.readers import InputExample

from torch.utils.data import DataLoader


def train_model(batch_size: int, epochs: int, train_examples: list[InputExample], model_path: str, sentences1: list[str], sentences2: list[str], scores: list[float]):
    """
    Train the model
    """

    evaluator = evaluation.BinaryClassificationEvaluator(sentences1, sentences2, scores, write_csv = True)


    # model_name = "thatdramebaazguy/roberta-base-MITmovie-squad"
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device='cuda')

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs = epochs,
        evaluator=evaluator,
        evaluation_steps = 2000,
        warmup_steps = math.ceil(len(train_dataloader) * 4 * 0.1), 
        output_path = model_path, 
        checkpoint_path = "../data/checkpoints/"
    )
        
    # model.save(model_path)
    
    return model


def encode_test_set(model, corpus: dict, queries: dict, test: pd.DataFrame, test_path: str):
    """
    With the finetuned model we will encode the test set and store it
    """
    count = 0
    test_data = []
    for index, row in test.iterrows():  
        query_id = row['query_id']; movie_id = row['movie_id']; label = float(row['label'])
        if query_id not in queries or movie_id not in corpus:
            count += 1
            continue
        else:
            query = queries[query_id]
            test_data.append(query)

    pool = model.start_multi_process_pool()
    test_embeddings = model.encode_multi_process(test_data, pool)

    with open(test_path, "wb") as fOut:
        pickle.dump({'sentences': test_data, 'embeddings': test_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    return test_embeddings    


def encode_corpus(corpus: dict, model_path: str, embedings_path: str):
    """
    Encode the corpus using SentenceTransformer
    """

    model = SentenceTransformer(model_path)
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(corpus.values(), pool)

    with open(embedings_path, "wb") as fOut:
        pickle.dump({'sentences': corpus, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


    return embeddings


def load_embeddings(embedings_path: str):
    """
    Load pre trained embeddings
    """
    
    with open(embedings_path, 'rb') as pkl:
        data = pickle.load(pkl)
        sentences = data['sentences']
        embeddings = data['embeddings']

    return embeddings, sentences