import pandas as pd
import numpy as np
import nltk

import string
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora
from gensim.models.ldamodel import LdaModel

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove tab, new line, and backslash
    text = text.replace('\\t', ' ').replace('\\n', ' ').replace('\\u', '').replace('\\', '')
    # Remove non ASCII characters
    text = text.encode('ascii', 'replace').decode('ascii')
    # Remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # Remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove leading and trailing whitespace
    text = text.strip()
    # Remove multiple whitespace into single whitespace
    text = re.sub('\s+', ' ', text)
    # Remove single characters
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    # NLTK word tokenize
    tokens = word_tokenize(text)
    return tokens

def stopword(text):
    # Get stopwords from NLTK for Indonesian
    list_stopwords = stopwords.words('indonesian')

    # Additional stopwords to append
    additional_stopwords = pd.read_csv("stopword_noise/stopword_noise.txt", sep=" ")

    # Extend the list of stopwords
    list_stopwords.extend(additional_stopwords)

    # Convert the list of stopwords to a set for faster lookup
    stopwords_set = list(set(list_stopwords))
    # Create Sastrawi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    #remove stopword
    tokens_without_stopwords = [word for word in text if word not in stopwords_set]
    #sastrawi
    # tweet_stem = stemmer.stem(tokens_without_stopwords)
    tweet_stem = [stemmer.stem(token) for token in tokens_without_stopwords]
    return tweet_stem


def create_lda_inputs(text):
    # Create a Gensim dictionary
    dictionary = corpora.Dictionary(text)
    # print(dictionary)

    # Generate the document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text]

    return dictionary, doc_term_matrix

def perform_lda(doc_term_matrix, total_topics, dictionary, number_words):
    lda_model = LdaModel(doc_term_matrix, num_topics=total_topics, id2word = dictionary, minimum_probability=0, random_state= 21,alpha= 'symmetric', eta='symmetric')
    topics = lda_model.show_topics(num_topics=total_topics, num_words=number_words)
    return topics

