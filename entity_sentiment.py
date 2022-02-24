import pandas as pd
from simple_nlp import get_entities
import sqlite3
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import pipeline
import re
from urlextract import URLExtract
import contractions
import nltk
connection = sqlite3.connect('news.db')
df = pd.read_sql_query("SELECT title , date , text FROM kaggle_news", connection)
df = df.head(20)
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
    #contractions
    text = contractions.fix(text)
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
    nlp = spacy.load('en_core_web_md')
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

      #sentiment model 
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
  qa_tokenizer = AutoTokenizer.from_pretrained("./models/qa_model")
qa_model = AutoModelForQuestionAnswering.from_pretrained("./models/qa_model")
qa_model = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

sent_tokenizer = AutoTokenizer.from_pretrained("./models/sentiment_model")
sent_model = AutoModelForSequenceClassification.from_pretrained("./models/sentiment_model")
sent_model = pipeline('text-classification', model=sent_model, tokenizer=sent_tokenizer)

def compute(text, aspects, qa_model, sent_model):
	#preprocessing
	preprocess_text = preprocess(text)
	#get nouns
	noun_list = get_noun(preprocess_text)	
	#get alternative names of aspects
	aspect_classes = get_similar_words(noun_list, aspects)
	#get sentiment
	sentiment_result = get_sentiment(aspect_classes, text, qa_model, sent_model)
	return sentiment_result



def aspect_sentiment(aspects, hashtag):
	# request_content = requests.get_json()
	# aspects = request_content.get('aspects', None)
	# hashtag = request_content.get('hashtag', None)

	if not aspects or aspects == list() :
		return {'statusCode': 400, 'body': 'aspects not found in request'}

	if not hashtag or hashtag.strip() == '':
		return {'statusCode': 400, 'body': 'hashtag not found in request'}

	#extracts 50 tweets regarding the hashtag from twitter
	twitter_content = get_tweets(hashtag, consumer_key, consumer_secret, access_token, access_token_secret, tweet_count=50)
	aspect_score = {asp : {'positive': 0, 'negative': 0} for asp in aspects}
	
	#compute scores for each tweet
	if twitter_content.to_dict():
		for text in twitter_content['text']:
			sentiment_result = compute(text, aspects, qa_model, sent_model)
			for result in sentiment_result:
				score = sentiment_result[result]
				if score>0:
					aspect_score[result]['positive'] = aspect_score[result]['positive'] + score
				elif score<0:
					aspect_score[result]['negative'] = aspect_score[result]['negative'] - score
				else:
					pass
	else:
		return {'statusCode': 400, 'body': 'No twitter data scraped for this hashtag'}



	result_list = [[k, 'positive', v['positive']] for k,v in aspect_score.items()]
	result_list.extend([[k, 'negative', v['negative']] for k,v in aspect_score.items()])

	#plot the bar plot across all aspects
	aspects_df = pd.DataFrame(result_list, columns= ['aspect', 'sentiment', 'score'])
	sns.barplot(x = 'aspect', y = 'score', hue='sentiment', data=aspects_df)
	plt.savefig('result.png')

	#send base64 string of image as response
	img_result = None
	with open('result.png', 'rb') as f:
		im_b64 = base64.b64encode(f.read())
		img_result = str(im_b64)
	if img_result:
		return {'statusCode': 200, 'body': json.dumps(img_result)}
	else:
		return {'statusCode': 400, 'body': 'issue in saving result image'}

qa_tokenizer = AutoTokenizer.from_pretrained("./models/qa_model")
qa_model = AutoModelForQuestionAnswering.from_pretrained("./models/qa_model")
qa_model = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
sent_tokenizer = AutoTokenizer.from_pretrained("./models/sentiment_model")
sent_model = AutoModelForSequenceClassification.from_pretrained("./models/sentiment_model")
sent_model = pipeline('text-classification', model=sent_model, tokenizer=sent_tokenizer)

def compute(text, aspects, qa_model, sent_model):
	#preprocessing
	preprocess_text = preprocess(text)
	#get nouns
	noun_list = get_noun(preprocess_text)	
	#get alternative names of aspects
	aspect_classes = get_similar_words(noun_list, aspects)
	#get sentiment
	sentiment_result = get_sentiment(aspect_classes, text, qa_model, sent_model)
	return sentiment_result



def aspect_sentiment(aspects, hashtag):

	if not aspects or aspects == list() :
		return {'statusCode': 400, 'body': 'aspects not found in request'}

	if not hashtag or hashtag.strip() == '':
		return {'statusCode': 400, 'body': 'hashtag not found in request'}

	#extracts 50 tweets regarding the hashtag from twitter
	twitter_content = get_tweets(hashtag, consumer_key, consumer_secret, access_token, access_token_secret, tweet_count=50)
	aspect_score = {asp : {'positive': 0, 'negative': 0} for asp in aspects}
	
	#compute scores for each tweet
	if twitter_content.to_dict():
		for text in twitter_content['text']:
			sentiment_result = compute(text, aspects, qa_model, sent_model)
			for result in sentiment_result:
				score = sentiment_result[result]
				if score>0:
					aspect_score[result]['positive'] = aspect_score[result]['positive'] + score
				elif score<0:
					aspect_score[result]['negative'] = aspect_score[result]['negative'] - score
				else:
					pass
	else:
		return {'statusCode': 400, 'body': 'No twitter data scraped for this hashtag'}



	result_list = [[k, 'positive', v['positive']] for k,v in aspect_score.items()]
	result_list.extend([[k, 'negative', v['negative']] for k,v in aspect_score.items()])

	#plot the bar plot across all aspects
	aspects_df = pd.DataFrame(result_list, columns= ['aspect', 'sentiment', 'score'])
	sns.barplot(x = 'aspect', y = 'score', hue='sentiment', data=aspects_df)
	plt.savefig('result.png')

	#send base64 string of image as response
	img_result = None
	with open('result.png', 'rb') as f:
		im_b64 = base64.b64encode(f.read())
		img_result = str(im_b64)
	if img_result:
		return {'statusCode': 200, 'body': json.dumps(img_result)}
	else:
		return {'statusCode': 400, 'body': 'issue in saving result image'}