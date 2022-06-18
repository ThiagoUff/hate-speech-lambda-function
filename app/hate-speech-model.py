import pandas as pd
import numpy as np
import nltk
import spacy
import pickle
import string

from sklearn.linear_model import LogisticRegression

#stop_words = spacy.lang.pt.stop_words.STOP_WORDS
punctuations = string.punctuation
nlp = spacy.load('pt_core_news_sm')

def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    #mytokens = [ word.lemma_.lower().strip() for word in mytokens ]

    # Removing stop words
    #mytokens = [word for word in mytokens if word not in stop_words]

    # Removing punctuations
    mytokens = [str(word) for word in mytokens if str(word) not in punctuations]
    # return preprocessed list of tokens
    return mytokens


def removeChars(dataFrame):
    # Remove @ tags
    dataFrame.txt = dataFrame.txt.str.replace(r'(@\w*)', '', regex=True)

    # Remove URL
    dataFrame.txt = dataFrame.txt.str.replace(r"http\S+", "", regex=True)

    # Remove # tag
    dataFrame.txt = dataFrame.txt.str.replace(r'#\w+', "", regex=True)
    # comp_df.tweet = comp_df.tweet.str.replace(r'#+',"")

    # Remove all non-character
    # comp_df.tweet = comp_df.tweet.str.replace(r"[^a-zA-Z ]","")

    # Remove extra space
    dataFrame.txt = dataFrame.txt.str.replace(r'( +)', " ", regex=True)
    dataFrame.txt = dataFrame.txt.str.strip()

    # Change to lowercase
    dataFrame.txt = dataFrame.txt.str.lower()


insults_df = pd.read_json('dataset/df_dataset.json')

removeChars(insults_df)
insults_df['corpus'] = [spacy_tokenizer(text) for text in insults_df.txt]

x_train = insults_df['corpus'].squeeze()
y_train = insults_df['has_anger'].squeeze()

#freq_vector = CountVectorizer(min_df=2, ngram_range=(1, 2)).fit(insults_df.corpus)

classifier = LogisticRegression(max_iter=500)

classifier.fit(x_train, y_train)

y_pred_train = classifier.predict(x_train)


pickle.dump( classifier, open( "pickled_model.p", "wb" ) )