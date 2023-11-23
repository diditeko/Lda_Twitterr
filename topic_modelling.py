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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




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

stemm_exceptions = ['pemilu', 'memerangi', 'Pemilu']
def stopword(text):
    # Get stopwords from NLTK for Indonesian
    list_stopwords = stopwords.words('indonesian')

    # Additional stopwords to append
    additional_stopwords = pd.read_csv("stopword/stopwords_noise.txt", sep=" ")

    # Extend the list of stopwords
    list_stopwords.extend(additional_stopwords)

    # Convert the list of stopwords to a set for faster lookup
    stopwords_set = list(set(list_stopwords))
    # Create Sastrawi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def custom_stem(word):
        return word in stemm_exceptions
    
    #remove stopword
    tokens_without_stopwords = [word for word in text if word not in stopwords_set]
    #sastrawi
    # tweet_stem = stemmer.stem(tokens_without_stopwords)
    # tweet_stem = [stemmer.stem(token) for token in tokens_without_stopwords]
    tweet_stem = [word if custom_stem(word) else stemmer.stem(word) for word in tokens_without_stopwords]
    return tweet_stem


def create_lda_inputs(text):
    # Create a Gensim dictionary
    dictionary = corpora.Dictionary(text)
    # print(dictionary)

    # Generate the document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text]

    return [dictionary, doc_term_matrix]

def perform_lda(doc_term_matrix, total_topics, dictionary, number_words):
    lda_model = LdaModel(doc_term_matrix, num_topics=total_topics, id2word = dictionary, minimum_probability=0, random_state= 21,alpha= 'symmetric', eta='symmetric')
    topics = lda_model.show_topics(num_topics=total_topics, num_words=number_words,formatted=False)
        # Extract words from the topics
    # formatted_topics = [{"topic_num": topic_num, "words": [word for word, prob in words]} for topic_num, words in topics]
    # formatted_topics = {str(topic_num): [word for word, prob in words] for topic_num, words in topics}
    formatted_topics = [{"topic_num": str(topic_num), "words": [word for word, prob in words]} for topic_num, words in topics]
    return lda_model, formatted_topics
    # return topics

def perform_tsne(lda_model, doc_term_matrix):
    # Create a matrix of topic contributions
    hm = np.array([[y for (x,y) in lda_model[doc_term_matrix[i]]] for i in range(len(doc_term_matrix))])
    # print(hm)
    # Convert to DataFrame and fill NaN values with 0
    arr = pd.DataFrame(hm).fillna(0).values
    # print(arr)
    scaler =StandardScaler()
    scaler_arr =  scaler.fit_transform(arr)
    # Perform t-SNE dimension reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=21, angle=.7, init='pca', perplexity=1)
    tsne_lda = tsne_model.fit_transform(scaler_arr)
    #coordinates
    x = tsne_lda[:, 0]
    y = tsne_lda[:, 1]
    coordinatess = pd.DataFrame({'x': x, 'y': y})

    return coordinatess

