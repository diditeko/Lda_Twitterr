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
from test2 import preprocess_text, stopword, perform_lda, create_lda_inputs, perform_tsne
from typing import List
from itertools import chain
import os
<<<<<<< HEAD
import logging
=======
>>>>>>> 122d9c77ee7ac90aaee40296d0d15e2f1de2651b
import shutil
import warnings
warnings.filterwarnings("ignore")

<<<<<<< HEAD

#set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


=======
>>>>>>> 122d9c77ee7ac90aaee40296d0d15e2f1de2651b
app = FastAPI()

# Load your LDA model and dictionary here
# Replace 'model_filename' and 'dictionary_filename' with your actual file paths
model_filename = "model\model\lda_model3"
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
<<<<<<< HEAD
    try:
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
        print(dictionary)
        

        n_samples = len(doc_term_matrix)
        print(n_samples)
        # Perform LDA on the combined corpus
        topics = perform_lda(doc_term_matrix, total_topics, dictionary, number_words)
        # print(topics)
        coordinates = perform_tsne(lda_model, doc_term_matrix)
        # Process the topic results and format them as needed
        # formatted_topics = [{"topic_num": i, "words": words} for i, words in enumerate(topics)]
        # print(formatted_topics)

        results = []
        for i, text in enumerate(texts):
            text_topics = lda_model[doc_term_matrix[i]]
        
            # Ensure that the values are explicitly converted to Python floats
            dominant_topic = max(text_topics, key=lambda x: float(x[1]))[0]
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
            # "Topic_Words": topics[dominant_topic]["words"]  # Include words for the dominant topic
            }
            results.append(result)

        logger.debug("Processed topic modeling request.")
        logger.info(f"topic: {topics},topic_data : {results}")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")
        # print(results)
        return {"topic": topics,"topic_data": results}

    except Exception as e :
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
=======
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
    print(dictionary)
    

    n_samples = len(doc_term_matrix)
    print(n_samples)
    # Perform LDA on the combined corpus
    topics = perform_lda(doc_term_matrix, total_topics, dictionary, number_words)
    # print(topics)
    coordinates = perform_tsne(lda_model, doc_term_matrix)
    # Process the topic results and format them as needed
    # formatted_topics = [{"topic_num": i, "words": words} for i, words in enumerate(topics)]
    # print(formatted_topics)

    results = []
    for i, text in enumerate(texts):
        text_topics = lda_model[doc_term_matrix[i]]
    
        # Ensure that the values are explicitly converted to Python floats
        dominant_topic = max(text_topics, key=lambda x: float(x[1]))[0]
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
        # "Topic_Words": topics[dominant_topic]["words"]  # Include words for the dominant topic
        }
        results.append(result)
    # print(results)
    return {"topic": topics,"topic_data": results}

>>>>>>> 122d9c77ee7ac90aaee40296d0d15e2f1de2651b

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6529)