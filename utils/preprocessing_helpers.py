from utils.fasttext_cleaner import fasttext_cleaning
import numpy as np

def small_cleaning_tweets(tokenized_text):
    new_tokenized_text = []
    for token in tokenized_text:
        if token.startswith('#') and len(token)>1:
            new_tokenized_text.append('#')
            new_tokenized_text.append(token[1:])
        elif token.startswith('@') and len(token)>1:
            new_tokenized_text.append('<TWEETER_USER>')
        else:
            new_tokenized_text.append(token)
    return new_tokenized_text

def _process_text(text, tweet):
    text = text.decode('utf-8')
    processed_text = dict()
    tokenized_text = fasttext_cleaning(text).split(' ')
    if tweet:
        tokenized_text = small_cleaning_tweets(tokenized_text)
    sequence_length = np.array(len(tokenized_text),dtype=np.int32)
    return ' '.join(tokenized_text).encode('utf-8'), sequence_length 
    

def process_text_with_label(text,label ,tweet=False):
    tokenized_text, sequence_length = _process_text(text, tweet)
    return tokenized_text, sequence_length, label

def process_text_without_label(text, tweet=False):
    return _process_text(text, tweet)