import sys
sys.settrace
import csv
from collections import defaultdict
from datetime import datetime as dt
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import threading


def read_text_file(file:str, col:str = 'clean_tweet_text', id_col:str='tweet_id'):
    columns = defaultdict(list)
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader: 
            if row[col] is None:
                continue
            columns[id_col].append(row[id_col])
            columns[col].append(row[col])

    return columns
            

def main(file = '../data/clean/features/text_EM_mini.csv'):
    RD_STATE = 12345
    print('Instantiating UMAP model')
    umap_model = UMAP(n_neighbors=15, n_components=5, 
                    low_memory=True,
                  min_dist=0.0, metric='cosine', random_state=RD_STATE)

    print('Instantiating vectorizer')
    vectorizer_model = CountVectorizer(
                                    # ngram_range = (1,1), 
                                    stop_words='english', 
                                    min_df=0.01, max_df=0.9)

    print('Instantiating topic model')
    topic_model = BERTopic(
        # embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        umap_model = umap_model,
        low_memory=True,
        #setting calc probs to false
        calculate_probabilities = False,
        #reducing the dimensionality via the vectorizer
        vectorizer_model = vectorizer_model

        )

    print('Reading in data')
    text_col = 'clean_tweet_text'
    # file = '../data/clean/features/text_EM.csv'
    
    data = read_text_file(file, text_col)
    docs = data[text_col]

    print('Fitting Model')
    topics, probs = topic_model.fit_transform(docs)
    topic_model.get_topic_info()
    print('Saving model output')
    topic_model.save("../models/topic/BertTopic/BTopic_model_", + dt.now().strftime("%d_%m_%Y %H_%M_%S") )

    return

if __name__ == '__main__':

    sys.setrecursionlimit(2097152)    # adjust numbers
    threading.stack_size(134217728)   # for your needs

    main_thread = threading.Thread(target=main)
    main_thread.start()
    main_thread.join()

    main()