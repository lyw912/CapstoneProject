# -*- coding: utf-8 -*-
import jieba
import re
import os
import pickle
from typing import List, Tuple, Any


# Load stopwords
stopwords = []
stopwords_path = "data/stopwords.txt"
if os.path.exists(stopwords_path):
    with open(stopwords_path, "r", encoding="utf8") as f:
        for w in f:
            stopwords.append(w.strip())
else:
    print(f"Warning: Stopwords file {stopwords_path} does not exist, will use empty stopwords list")


def load_corpus(path):
    """
    Load corpus
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing(content)
            data.append((content, int(seniment)))
    return data


def load_corpus_bert(path):
    """
    Load corpus for BERT
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing_bert(content)
            data.append((content, int(seniment)))
    return data


def processing(text):
    """
    Data preprocessing, can be overridden according to your needs
    """
    # Data cleaning
    text = re.sub("\{%.+?%\}", " ", text)           # Remove {%xxx%} (geo-location, Weibo topics, etc.)
    text = re.sub("@.+?( |$)", " ", text)           # Remove @xxx (username)
    text = re.sub("【.+?】", " ", text)              # Remove 【xx】 (content inside is usually not written by user)
    text = re.sub("\u200b", " ", text)              # '\u200b' is a bad case in this dataset, not important
    # Tokenization
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    # Special handling for negation word `不`: concatenate with following word
    while "不" in words:
        index = words.index("不")
        if index == len(words) - 1:
            break
        words[index: index+2] = ["".join(words[index: index+2])]  # Cool list slice assignment
    # Join with spaces
    result = " ".join(words)
    return result


def processing_bert(text):
    """
    Data preprocessing for BERT, can be overridden according to your needs
    """
    # Data cleaning
    text = re.sub("\{%.+?%\}", " ", text)           # Remove {%xxx%} (geo-location, Weibo topics, etc.)
    text = re.sub("@.+?( |$)", " ", text)           # Remove @xxx (username)
    text = re.sub("【.+?】", " ", text)              # Remove 【xx】 (content inside is usually not written by user)
    text = re.sub("\u200b", " ", text)              # '\u200b' is a bad case in this dataset, not important
    return text


def save_model(model: Any, model_path: str) -> None:
    """
    Save model to file
    
    Args:
        model: Model object to save
        model_path: Save path
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")


def load_model(model_path: str) -> Any:
    """
    Load model from file
    
    Args:
        model_path: Model file path
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded: {model_path}")
    return model


def preprocess_text_simple(text: str) -> str:
    """
    Simple text preprocessing function for text cleaning during prediction
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Data cleaning
    text = re.sub("\{%.+?%\}", " ", text)           # Remove {%xxx%}
    text = re.sub("@.+?( |$)", " ", text)           # Remove @xxx
    text = re.sub("【.+?】", " ", text)              # Remove 【xx】
    text = re.sub("\u200b", " ", text)              # Remove special characters
    
    # Remove emojis
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f018-\U0001f270\U0000231a-\U0000231b\U0000238d-\U0000238d\U000024c2-\U0001f251]+', '', text)
    
    # Merge multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()