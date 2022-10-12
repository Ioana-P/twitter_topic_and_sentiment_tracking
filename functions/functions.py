## principal functions and objects file

# clear sections are shown in comments
# go to docstrings for function purpose and arguments

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import silhouette_score
import string

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
# stop_words += ['__', '___']

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

from functions.nlp_eda import *
from pathlib import Path



#################################CLEANING#####################################


def clean_tweet_text(raw:Path, text_col_raw:str='tweet_text')->tuple:
    """Wrapper fn for NLProc cleaning functions inside nlp_eda

    Args:
        raw (Path): path to raw data csv
        text_col_raw (str, optional): Column containing tweets. Defaults to 'tweet_text'.

    Returns:
        tuple: (clean dataframe, scikit-learn count vectorizer object)
    """    

    clean = etl_tweet_text(raw, text_col_raw)

    text_col_clean = f'clean_{text_col_raw}'

    clean_cvt, cvt = get_count_vectorized_df(clean, text_col_clean)

    clean = clean.join(clean_cvt)

    return clean, cvt


def is_it_before_or_after(x, date):
    try:
        if x<=date:
            return 'Before'
        else:
            return 'After'
    except: # (ValueError, ParserError):
        return np.NAN

#################################API-REQUESTS#####################################




#################################SCRAPING#####################################





#################################DATA TRANSFORMATION#####################################



#################################EDA#####################################



#################################SUMMARY TABLES CREATION#####################################



#############################MODEL BUILDING, GRIDSEARCH AND PIPELINES#####################################
def get_tweet_as_embed(token_lst:list, model):

    try:
        sub_vects = []
        for token in token_lst:
            try:
                sub_vects.append(model.wv.get_vector(token.lower()))
            except:
                continue

        # adding and normalising the sub vectors
        # print(sub_vects)
        sum_vect = sub_vects[0]
        for vec in sub_vects[1:]:
            sum_vect = np.add(sum_vect, vec)

        sum_vect = np.divide(sum_vect, len(sub_vects))
        return sum_vect
    except Exception as E:
        # print(E)
        return np.NAN


#############################MODEL EVALUATION (METZ, ROC CURVE, CONF_MAT)#####################################

