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
from topic_modelling import preprocess_text, stopword, perform_lda, create_lda_inputs, perform_tsne
from typing import List
from itertools import chain
import os
import shutil
import warnings
import logging
warnings.filterwarnings("ignore")


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

@app.post("/topic-modeling/")
async def perform_topic_modeling(data: InputData):
    texts = data.texts
    # print(texts)

    # combined_stem = stopword(tokens)    
    preproced_text = []
    for text in texts :
        tokens = preprocess_text(text)
        combined_stem = stopword(tokens)

        preproced_text.append(combined_stem)
        # print(preproced_text)

    
    # Create LDA inputs for the combined corpus
    dictionary, doc_term_matrix = create_lda_inputs(preproced_text)
    # print(dictionary)
    print(doc_term_matrix)
    

    n_samples = len(doc_term_matrix)
    print(n_samples)
    # Perform LDA on the combined corpus
    lda_model, topics = perform_lda(doc_term_matrix, total_topics, dictionary, number_words)
    print(topics)
    coordinates = perform_tsne(lda_model, doc_term_matrix)
    test = lda_model.show_topics(num_topics=total_topics, num_words=number_words,formatted=False)
    print(test)
    results = []
    for i, text in enumerate(texts):
        text_topics = lda_model[doc_term_matrix[i]]
        print(text_topics)
        # Ensure that the values are explicitly converted to Python floats
        dominant_topic = max(text_topics, key=lambda x: float(x[1]))[0]
        print(dominant_topic)
        # dominant_topic = max(text_topics, key=lambda x: x[1])[0]
        topic_perc_contrib = float(max(text_topics, key=lambda x: float(x[1]))[1])


        # Find the corresponding t-SNE coordinates
        coord_index = i if i < len(coordinates) else -1
        x_coord = float(coordinates['x'][coord_index])
        y_coord = float(coordinates['y'][coord_index])

        result = {
        "x" : x_coord,
        "y" : y_coord,
        "Dominant_Topic": dominant_topic,
        "Topic_Perc_Contrib": topic_perc_contrib,
        "Text": text
        }
        results.append(result)
    return {"topic": topics,"topic_data": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6529)