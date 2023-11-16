from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from main import preprocess_text, stopword, perform_lda, create_lda_inputs

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
    text: str
    # num_topics: int
    # num_topics: int = default_num_topics

# Define an API route to perform topic modeling
@app.post("/topic-modeling/")
async def perform_topic_modeling(data: InputData):
    text = data.text
    # Preprocess the input text (similar to what you did in your code)
    # ...

    # Tokenize and preprocess the input text
    tokens = preprocess_text(text)
    tweet_stem = stopword(tokens)
    # Convert the text to a bag of words
    # bow = dictionary.doc2bow(tweet_stem)

    num_topics = total_topics
    lda_model.num_topics = num_topics
    dictionary, doc_term_matrix = create_lda_inputs([tweet_stem])

    # Get the dominant topic for the input text
    # topics = perform_lda(bow, num_topics, dictionary, 5)
    topics = perform_lda(doc_term_matrix, num_topics, dictionary, number_words)

    

    # Process the topic results and format them as needed
    formatted_topics = [{"topic_num": topic_num, "words": words} for topic_num, words in topics]

    return {"topics": topics}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
