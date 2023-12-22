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
from topic_test import preprocess_text, stopword,perform_lda, create_lda_inputs, perform_tsne
from typing import List
from itertools import chain
import os
from openTSNE import TSNE 
import shutil
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")


app = FastAPI()

# Load your LDA model and dictionary here
model_filename = "model\model\lda_model3"
# lda_model =None



lda_model = LdaModel.load(model_filename)
total_topics = 5 # jumlah topik yang akan di extract
number_words = 3

# Create a Pydantic model for input data
class InputData(BaseModel):
    texts: List[str]

class TSNEInputData(BaseModel):
    texts: List[str]

@app.post("/topic-modeling_multithread/")
async def perform_topic_modeling(data: InputData):
    start_time = time.time()
    texts = data.texts

    # preproced_text = []
    # for text in texts :
    #     tokens = preprocess_text(text)
    #     combined_stem = stopword(tokens)

    #     preproced_text.append(combined_stem)
    with ThreadPoolExecutor(max_workers=4) as executor:
        preprocessed_texts = list(executor.map(preprocess_text, texts))
    with ThreadPoolExecutor(max_workers=4) as executor:
        stopped = list(executor.map(stopword, preprocessed_texts))
    end_time = time.time()  # Record the end time
    duration = end_time - start_time
    print(duration)

    dictionary, doc_term_matrix = create_lda_inputs(stopped)

    # # Parallelize the LDA computation

    with ThreadPoolExecutor(max_workers=4) as executor:
        lda_model, topics = next(executor.map(lambda x: perform_lda(*x), [(doc_term_matrix, total_topics, dictionary, number_words)]))


    # Parallelize the t-SNE computation

    with ThreadPoolExecutor(max_workers=4) as executor:
        # print(executor._max_workers)
        coordinate_generator = executor.map(lambda x: perform_tsne(*x), [(lda_model, doc_term_matrix)])
        while True:
            try:
                coordinates = next(coordinate_generator)
                # Perform any operation on coordinates
            except StopIteration:
                break
    # coordinates = perform_tsne(lda_model, doc_term_matrix)
    # print(coordinates)
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

if __name__ == "__main__":
    import uvicorn
    # load_lda_model()

    uvicorn.run(app, host="127.0.0.1", port=1501)