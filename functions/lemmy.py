from nltk.corpus import stopwords
from nltk import FreqDist, text
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string

tokenizer = RegexpTokenizer(r'\b[A-Za-z0-9\-]{2,}\b')
default_tk = tokenizer

def generate_stop_words_by_language(lst_languages:list)->list:
    """Returns a list with stopwords from different languages, based on 
    nltk.corpus.stopwords. Also adds punctuations to the list of stopwords. 
    Result to be used in data cleaning/omitting terms

    Args:
        lst_languages (list): list of languages to use. If an incorrect language 
        is specified, then it will be omitted but an error will NOT be raised

    Returns:
        list: list of stopwords 
    """    

    gen_stop_words = []
    for lang in lst_languages:
        try:
            gen_stop_words += list(set(stopwords.words(lang)))

        except OSError: 
            print(f'Could not find stop words for {lang}. Omitting from list of stopwords')
            pass

    gen_stop_words += list(string.punctuation)

    return gen_stop_words

lst_languages = ['english', 'french', 'spanish', 'german', 'spanish', 'russian']
gen_stop_words = generate_stop_words_by_language(lst_languages)

gen_stop_words.extend(['amp', '&amp', '’', '&amp;'])

class LemmaTokenizer(object):
    def __init__(self, tokenizer = default_tk, stopwords = gen_stop_words):
        self.wnl = WordNetLemmatizer()
        self.tokenizer = tokenizer
        self.stopwords = stopwords
    def __call__(self, articles):
        return [self.wnl.lemmatize(token, ) for token in self.tokenizer.tokenize(articles) if token not in self.stopwords]
    
    def tokenize(self, articles):
        return [self.wnl.lemmatize(token) for token in self.tokenizer.tokenize(articles) if token not in self.stopwords]
    
