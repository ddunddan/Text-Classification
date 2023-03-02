import os
import os.path
import pandas as pd
import pickle

from transformers import BertTokenizer
from utils import make_id_file, make_id_file_test, SentimentDataset, SentimentTestDataset, collate_fn_style, collate_fn_style_test


def save_pickle(tokenizer):
    results = make_id_file('yelp', tokenizer)
    file_names = ['train_pos', 'train_neg', 'dev_pos', 'dev_neg']
    
    for i, v in enumerate(results):
        filename = f"data/{file_names[i]}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(v, f)
            
def load_pickle():
    file_names = ['train_pos', 'train_neg', 'dev_pos', 'dev_neg']
    temp = []
    for filename in file_names:
        with open(f'data/{filename}.pkl', 'rb') as f:
            temp.append(pickle.load(f))
            
    return temp
    
def data2dataset():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if not os.path.isfile('data/train_pos.pkl'):
        save_pickle(tokenizer)
        
    train_pos, train_neg, dev_pos, dev_neg = load_pickle()
    
    train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
    dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)
    
    test_df = pd.read_csv('data/test_no_label.csv')
    test_dataset = test_df['Id']
    test = make_id_file_test(tokenizer, test_dataset)
    test_dataset = SentimentTestDataset(tokenizer, test)

    return train_dataset, dev_dataset, test_dataset, test_df