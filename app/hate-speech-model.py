import pandas as pd
import pickle
import boto3
import json

#from sklearn.linear_model import LogisticRegression
#from sklearn.feature_extraction.text import CountVectorizer


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
s3_bucket = 'hatespeech-ml'
model_name = 'pickled_model.p'
vector_name = 'vector.pickel'
model_file_path = '/tmp/' + model_name
vector_file_path = '/tmp/' + vector_name