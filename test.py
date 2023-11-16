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
from topic_modelling import preprocess_text, stopword, perform_lda, create_lda_inputs , perform_tsne, format_topics_sentencesfix, format_topics_sentencesx
from typing import List
from itertools import chain
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# Load your LDA model and dictionary here
# Replace 'model_filename' and 'dictionary_filename' with your actual file paths
model_filename = "model\lda_model2"
# dictionary_filename = "dictionary\lda_v1.dict"

lda_model = LdaModel.load(model_filename)
# dictionary = Dictionary.load(dictionary_filename)
# default_num_topics = 5
total_topics = 5 # jumlah topik yang akan di extract
number_words = 5

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
    

    n_samples = len(doc_term_matrix)
    # print(n_samples)
    # Perform LDA on the combined corpus
    topics = perform_lda(doc_term_matrix, total_topics, dictionary, number_words)
    # print(topics)

    # Process the topic results and format them as needed
    formatted_topics = [{"topic_num": i, "words": words} for i, words in enumerate(topics)]
    print(formatted_topics) 


    # Create the desired output format
    output_topics = {str(topic["topic_num"]): topic["words"] for topic in formatted_topics}
    # print(output_topics)
    coordinates = perform_tsne(lda_model, doc_term_matrix)
    dominant_topics = format_topics_sentencesfix(lda_model, doc_term_matrix,dictionary,number_words,texts,output_topics)
    # dominant_topics['Perc_Contribution'] = dominant_topics['Perc_Contribution'].astype(str)
    dominant_topics_dict = dominant_topics.to_dict(orient='records')



    return {"topic": formatted_topics, "coordinates": coordinates.to_dict(orient='records'),"topics_data" :  dominant_topics_dict}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6529)