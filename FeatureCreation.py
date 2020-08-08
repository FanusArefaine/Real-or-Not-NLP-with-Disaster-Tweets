
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string

def word_count(df):
    df['word_count'] = [len(i.split()) for i in df['text']]
    return df


def unique_word_count(df):
    df['unique_word_count'] = df['text'].apply(lambda s: len(set(str(s).split())))
    return df


def character_count(df):
    df['char_count'] = df['text'].apply(lambda s: len(str(s)))
    return df

def stopwords_count(df):

   df['stop_word_count'] = df['text'].apply(lambda s: len([k for k in str(s).lower().split() if k in stopwords.words('english')]))
   return df

    

def punctuation_count(df):
    df['punctuation_count'] = df['text'].apply(lambda s: len([k for k in str(s) if k in string.punctuation]))
    return df
    

def mean_word_length(df):
    df['mean_word_length'] = df['text'].apply(lambda s: np.mean([len(w) for w in str(s).split()]))
    return df


def mention_count(df):
    df['mention_count'] = df['text'].apply(lambda s: len([k for k in str(s) if k=='@']))
    return df

def url_count(df):
    df['URL_count'] = df['text'].apply(lambda s: len([k for k in str(s).lower().split() if 'https' in k or 'http' in k]))
    return df

def hashtag_count(df):
    df['hashtag_count'] = df['text'].apply(lambda s: len([k for k in str(s) if k == '#']))
    return df



















