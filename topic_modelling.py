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
    topics = lda_model.show_topics(num_topics=total_topics, num_words=number_words)
    print(topics)
    return topics

def perform_tsne(lda_model, doc_term_matrix):
    # Create a matrix of topic contributions
    hm = np.array([[y for (x,y) in lda_model[doc_term_matrix[i]]] for i in range(len(doc_term_matrix))])
    print(hm)
    # Convert to DataFrame and fill NaN values with 0
    arr = pd.DataFrame(hm).fillna(0).values
    print(arr)
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(arr)
    # Perform t-SNE dimension reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=21, angle=.8, init='pca')
    tsne_lda = tsne_model.fit_transform(scaled_arr)
    #coordinates
    x = tsne_lda[:, 0]
    y = tsne_lda[:, 1]
    coordinatess = pd.DataFrame({'X': x, 'Y': y})

    return coordinatess

def format_topics_sentences(lda_model, doc_term_matrix, dictionary, number_words,texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[doc_term_matrix]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # print(row)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num, topn=number_words)
                # print(wp)
                topic_keywords = ", ".join([word for word, prop in wp])
                # print(topic_keywords)
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
                
    

    # Return the DataFrame
    contents = pd.Series(texts)
    print(contents)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    print(sent_topics_df)
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text']
    return sent_topics_df


def format_topics_sentences1(lda_model, doc_term_matrix, dictionary, number_words, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    for i, dtm in enumerate(doc_term_matrix):
        new_doc_term_matrix = dtm  # Use the provided doc_term_matrix

        # Get the dominant topic and its keywords from the processed text
        dominant_topic = pd.DataFrame()
        for row_list in lda_model[new_doc_term_matrix]:
            row = row_list[0] if lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = lda_model.show_topic(topic_num, topn=number_words)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    dominant_topic = dominant_topic.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break

        dominant_topic['Text'] = texts[i]  # Add the text to the DataFrame

        sent_topics_df = sent_topics_df.append(dominant_topic, ignore_index=True)

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text']
    return sent_topics_df

def format_topics_sentences4(lda_model, doc_term_matrix, dictionary, number_words):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[doc_term_matrix]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num, topn=number_words)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
                
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Return the DataFrame
    return sent_topics_df

def format_topics_sentencesx(lda_model, doc_term_matrix, dictionary, number_words, texts):
    data = []

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[doc_term_matrix]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Temporary variables for the dominant topic information
        dominant_topic = None
        perc_contribution = None
        topic_keywords = None

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # Consider only the dominant topic
                wp = lda_model.show_topic(topic_num, topn=number_words)
                topic_keywords = ", ".join([word for word, prop in wp])
                dominant_topic = int(topic_num)
                perc_contribution = round(prop_topic, 4)
                break  # No need to continue if dominant topic found

        # Append data to the list
        data.append([dominant_topic, perc_contribution, topic_keywords, texts[i]])

    # Create a DataFrame from the accumulated data
    sent_topics_df = pd.DataFrame(data, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text'])
    return sent_topics_df

def format_topics_sentencesfix(lda_model, doc_term_matrix, dictionary, number_words, texts, output_topics):
    data = []

    for i, row_list in enumerate(lda_model[doc_term_matrix]):
        row = row_list[0] if lda_model.per_word_topics else row_list

        # Temporary variables for the dominant topic information
        dominant_topic = None
        print(dominant_topic)
        perc_contribution = None
        topic_keywords = None
        print(topic_keywords)

        # Extract topic keywords from the output_topics based on the topic number
        for j, (topic_num, prop_topic) in enumerate(row):
            print(topic_num)
            if j == 0:  # Consider only the dominant topic
                topic_keywords = output_topics.get(str(topic_num), "No Keywords Found")
                # print(topic_keywords)
                dominant_topic = int(topic_num)
                # print(dominant_topic)
                perc_contribution = round(prop_topic, 4)
                break  # No need to continue if dominant topic found

        # Append data to the list
        data.append([dominant_topic, perc_contribution, topic_keywords, texts[i]])

    # Create a DataFrame from the accumulated data
    sent_topics_df = pd.DataFrame(data, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text'])
    return sent_topics_df


