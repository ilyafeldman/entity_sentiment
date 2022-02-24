from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import pipeline
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_model = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sent_model = pipeline('text-classification', model=sent_model, tokenizer=sent_tokenizer)
import spacy
import pandas as pd
import re
import nltk

def get_noun(lines):
	tokenized = nltk.word_tokenize(lines)
	nouns = set([word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')])
	return nouns

def preprocess(text):
    text = str(text).lower()
    #remove emails
    text = re.sub(r'\S*@\S*\s?',' ',text)
    #remove mentions
    text = re.sub(r'@\S+', ' ', text)
    #remove hashtags
    text = re.sub(r'@\S+', ' ', text)
    #remove emojis
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    #remove all punct
    text = re.sub('[^A-z0-9]', ' ', text)
    #remove extras whitespaces
    text = re.sub(' +', ' ', text)
    return text

def get_similar_words(nouns, aspects):
    nlp = spacy.load('en_core_web_sm')
    aspect_classes = {k: list() for k in aspects}
    for noun in nouns:
        scores = list()
    for aspect in aspects:
        aspect_token = nlp(aspect)
        noun_token = nlp(noun)
        similarity_score = aspect_token.similarity(noun_token)
        scores.append(similarity_score)
    index = scores.index(max(scores))
    aspect_name = aspects[index]

    if max(scores)>0.60:
        value = aspect_classes[aspect_name]
        value.append(noun)
        aspect_classes[aspect_name] = list(set(value))
    else:
        pass
    return aspect_classes	


def get_sentiment(aspect_classes, text):
    sentiment_dict = {k:0 for k in aspect_classes}
    for aspect in aspect_classes:
        alt_names = aspect_classes[aspect]
        for name in alt_names:
            question = f'how is {name}'
            QA_input = {'question': question, 'context': text}
            qa_result = qa_model(QA_input)
            answer = qa_result['answer']
            sent_result = sent_model(answer)    
            print(sent_result)
            sentiment = sent_result[0]['label']

            if sentiment == 'LABEL_0':
                sentiment, score = 'Negative', -1
            elif sentiment == 'LABEL_1':
                sentiment, score = 'Neutral', 0
            else:
                sentiment, score = 'Positive', 1

        value = sentiment_dict[aspect] + score
        sentiment_dict[aspect] = value
    return sentiment_dict


def compute(text, aspects):
	
    preprocess_text = preprocess(text)
    print(preprocess_text)
    noun_list = get_noun(preprocess_text)
    print(noun_list)
    
    aspect_classes = get_similar_words(noun_list, aspects)
    print(aspect_classes)
	
    sentiment_result = get_sentiment(aspect_classes, text)
    print(sentiment_result)
	
    return sentiment_result

def aspect_sentiment(aspects, df):
    aspect_score = {asp : {'positive': 0, 'negative': 0} for asp in aspects}
    if df.to_dict():
        for text in df['text']:
            sentiment_result = compute(text, aspects)
            for result in sentiment_result:
                score = sentiment_result[result]
                if score>0:
                    aspect_score[result]['positive'] = aspect_score[result]['positive'] + score
                elif score<0:
                    aspect_score[result]['negative'] = aspect_score[result]['negative'] - score
                else:
                    pass
    result_list = [[k, 'positive', v['positive']] for k,v in aspect_score.items()]
    result_list.extend([[k, 'negative', v['negative']] for k,v in aspect_score.items()])
    aspects_df = pd.DataFrame(result_list, columns= ['aspect', 'sentiment', 'score'])
    return aspects_df

# aspects = ['']
# aspect_sentiment(aspects , df)