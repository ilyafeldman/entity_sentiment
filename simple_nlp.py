# -*- coding: UTF-8 -*-
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
import pandas as pd
import spacy
from bertopic import BERTopic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import trange

def cleaning(df):
    '''Cleans the provided dataframe before usage'''
    df.text = df.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
    df.text = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
    df.drop_duplicates(subset='text' , inplace=True)
    print('cleaned')
    return df

def topics(df):
    '''BERTopic analysis. Returns a dataframe with topic_id and topic_description columns, as well as a graph containing topics over time usage'''
    cleaned_df = cleaning(df)
    timestamps = df.date.to_list()
    texts = df.text.to_list()
    topic_model = BERTopic(verbose=True , nr_topics="auto")
    topics , probs = topic_model.fit_transform(texts)
    topic_descriptions = []
    for topic in topics:
        description = topic_model.get_topic(topic)
        topic_descriptions.append(description)
    print('Building topics over time graph')
    topics_over_time = topic_model.topics_over_time(texts, topics, timestamps, nr_bins=20)
    cleaned_df = cleaned_df.assign(topic_id = topics)
    cleaned_df = cleaned_df.assign(topic_description = topic_descriptions)
    print('done topics')
    return cleaned_df , topics_over_time , topic_model

def entities(df):
    cleaned_df = cleaning(df)
    nlp = spacy.load("en_core_web_sm")
    total1 = []
    total2 = []
    for row in trange(cleaned_df['text'] , desc='Rows'):
        doc = nlp(row)
        ents_per_doc = []
        ent_w_descr_per_doc = []
        for entity in trange(doc.ents , desc='Entities'):
            ents_per_doc.append(entity)
            ent_w_descr = entity.text + '-' + spacy.explain(entity.label_)
            ent_w_descr_per_doc.append(ent_w_descr)
        total1.append(ents_per_doc)
        total2.append(ent_w_descr_per_doc)     
    cleaned_df = cleaned_df.assign(entities = total1)
    cleaned_df = cleaned_df.assign(entities_with_description = total2)
    print('done entities')
    return cleaned_df

def sentiment(df):
    cleaned_df = cleaning(df)
    analyzer = SentimentIntensityAnalyzer()
    sentims = []
    for row in cleaned_df['text']:
        sentim = analyzer.polarity_scores(row)
        sentims.append(int(sentim['compound']*100))
    cleaned_df = cleaned_df.assign(text_sentiment = sentims)
    print('done sentiment')
    return cleaned_df

def read_article(string):
    article = string.split(". ")
    sentences = [sentence.replace("[^a-zA-Z]", " ").split(" ") for sentence in article]
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def summary(text, summary_len=3):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences =  read_article(text)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    for i in range(summary_len):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    summarize_text = '. '.join(summarize_text)
    return summarize_text