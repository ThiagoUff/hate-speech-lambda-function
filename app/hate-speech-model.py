import pandas as pd
import numpy as np
import math
import spacy
import pickle
import string
import boto3
import os
import json
import re

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

#stop_words = spacy.lang.pt.stop_words.STOP_WORDS
punctuations = string.punctuation
nlp = spacy.load('pt_core_news_sm')

def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [str(word) for word in mytokens if str(word) not in punctuations]
    return " ".join(mytokens)

def removeChars(sentence):
    sentence = re.sub(r'(@\w*)', '', sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub(r'#\w+', "", sentence)
    sentence = re.sub(r'( +)', " ", sentence)
    sentence = sentence.strip()
    sentence = sentence.lower()

def lambda_handler(event, context):
    # Parse input
    body = event['body']
    input = json.loads(body)['data']
    input = str(input) 
   

    # Download pickled model from S3 and unpickle
    s3.download_file(s3_bucket, model_name, model_file_path)
    s3.download_file(s3_bucket, vector_name, vector_file_path)
   
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f) 
        
    with open(vector_file_path, 'rb') as f:
        freq_vector = pickle.load(f)  
   
    removeChars(input)
    input = spacy_tokenizer(input)
    d = {'corpus': [input]}
    df = pd.DataFrame(data=d)
    
    input = freq_vector.transform(df.corpus)

    prediction = model.predict(input)[0]   
    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": str(prediction),
        }),
    }
    
s3 = boto3.client('s3')
s3_bucket = os.environ['s3_bucket']
model_name = os.environ['model_name']
vector_name = os.environ['vector_name']
model_file_path = '/tmp/' + model_name
vector_file_path = '/tmp/' + vector_name