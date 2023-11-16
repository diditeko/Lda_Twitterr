from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from topic_modelling import preprocess_text, stopword, perform_lda, create_lda_inputs , perform_tsne
from typing import List
import os
import shutil


app = FastAPI()

# Load your LDA model and dictionary here
# Replace 'model_filename' and 'dictionary_filename' with your actual file paths
model_filename = "model\lda_model2"
dictionary_filename = "dictionary\lda_v1.dict"

lda_model = LdaModel.load(model_filename)
dictionary = Dictionary.load(dictionary_filename)
# default_num_topics = 5
total_topics = 5 # jumlah topik yang akan di extract
number_words = 5

# Create a Pydantic model for input data
class InputData(BaseModel):
    texts: List[str]


# Define an API route to perform topic modeling
@app.post("/topic-modeling/")
async def perform_topic_modeling(data: InputData):
    texts = data.texts
    # topics_list = []
    # tsne_coordinates_list = []


    for text in texts:
        # Tokenize and preprocess the input text
        tokens = preprocess_text(text)
        tweet_stem = stopword(tokens)

        num_topics = total_topics
        lda_model.num_topics = num_topics
        dictionary = dictionary_filename
        dictionary, doc_term_matrix = create_lda_inputs([tweet_stem])
        
        

        # Perform LDA topic modeling
        topics = perform_lda(doc_term_matrix, num_topics, dictionary, number_words)

        n_samples = len(doc_term_matrix)
        print("Number of samples:", n_samples)

        # #Perform t-SNE dimension reduction and obtain coordinates
        # tsne_coordinates = perform_tsne(lda_model, doc_term_matrix)

        # topics_list.append(topics)
        # tsne_coordinates_list.append(tsne_coordinates.to_dict(orient='records'))

    return {"topics": topics}


    

    # return {"topics": topics, "tsne_coordinates": tsne_lda.to_dict(orient='records')}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
