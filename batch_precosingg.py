from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from topic_multithreadlda import preprocess_text, stopword, perform_lda, create_lda_inputs, perform_tsne
from typing import List
from itertools import chain
import os
import shutil
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
import asyncio


app = FastAPI()

# Load your LDA model and dictionary here
model_filename = "model\model\lda_model3"


lda_model = LdaModel.load(model_filename)
total_topics = 5 # jumlah topik yang akan di extract
number_words = 3

# Create a Pydantic model for input data
class InputData(BaseModel):
    texts: List[str]

class TSNEInputData(BaseModel):
    texts: List[str]

async def perform_modelling_batch (texts):
        # Preprocess all texts at once
    preproced_text = []
    for text in texts :
        tokens = preprocess_text(text)
        combined_stem = stopword(tokens)

        preproced_text.append(combined_stem)

    dictionary, doc_term_matrix = create_lda_inputs(preproced_text)

    with ThreadPoolExecutor(max_workers=8) as executor:
        lda_model, topics = next(executor.map(lambda x: perform_lda(*x), [(doc_term_matrix, total_topics, dictionary,  number_words)]))

    with ThreadPoolExecutor(max_workers=4) as executor:
        # print(executor._max_workers)
        coordinate_generator = executor.map(lambda x: perform_tsne(*x), [(lda_model, doc_term_matrix)])
        while True:
            try:
                coordinates = next(coordinate_generator)
                # Perform any operation on coordinates
            except StopIteration:
                break

    results = []
    for i, text in enumerate(texts):
        text_topics = lda_model[doc_term_matrix[i]]
        dominant_topic = max(text_topics, key=lambda x: float(x[1]))[0]
        topic_perc_contrib = float(max(text_topics, key=lambda x: float(x[1]))[1])

        coord_index = i if i < len(coordinates) else -1
        # print(coord_index)
        x_coord = float(coordinates['x'][coord_index])
        y_coord = float(coordinates['y'][coord_index])

        result = {
            "x": x_coord,
            "y": y_coord,
            "Dominant_Topic": dominant_topic,
            "Topic_Perc_Contrib": topic_perc_contrib,
            "Text": text
        }
        results.append(result)

    return {"topic": topics, "topic_data": results}

@app.post("/topic-modeling_multithread/")
async def perform_topic_modeling_endpoint(data: InputData):
    texts = data.texts

    # Split texts into batches (adjust batch size based on your needs)
    batch_size = 1000
    text_batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    # Process batches asynchronously
    results = await asyncio.gather(*(perform_modelling_batch(batch) for batch in text_batches))

    # Combine results from batches
    combined_results = {
        "topic": results[0]["topic"],  # Assuming topics are the same for all batches
        "topic_data": [result for batch_result in results for result in batch_result["topic_data"]]
    }

    return combined_results

if __name__ == "__main__":


    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6529)