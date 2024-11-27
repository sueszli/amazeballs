import pandas as pd
import json
import copy
import hashlib
import json
import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from tqdm import tqdm
from utils import *
import datasets 
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments
from transformers import pipeline

datasets.logging.set_verbosity_error()
set_env()

data_path = get_current_dir().parent / "data"
dataset_path = get_current_dir().parent / "datasets"
weights_path = get_current_dir().parent / "weights"
output_path = get_current_dir()

os.makedirs(data_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)


# 
# data
# 


def get_asin2category():
    # use `parent_asin` key from metadata to look up the category of each reviewed item
    file_path = hf_hub_download(repo_id="McAuley-Lab/Amazon-Reviews-2023", filename="asin2category.json", repo_type="dataset", cache_dir=dataset_path)
    file = open(file_path, "r")
    data = json.load(file)
    file.close()
    return data

def get_category_metadata(category):
    data = datasets.load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", cache_dir=dataset_path, trust_remote_code=True)
    return data

def get_all_categories():
    data = datasets.load_dataset("text", data_files="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/all_categories.txt", streaming=False, cache_dir=dataset_path, trust_remote_code=True)
    return data["train"].to_dict()["text"]

def get_category_data(category, sample_size):
    # cache hit
    cachepath = dataset_path / f"cache_{category}_{sample_size}.csv"
    if cachepath.exists():
        data = pd.read_csv(cachepath)
        return data

    # cache miss
    data = datasets.load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", streaming=True, cache_dir=dataset_path, trust_remote_code=True)
    sampled_data = []
    for entry in tqdm(data["full"].shuffle(seed=42).take(sample_size), total=sample_size, desc=f"sampling {category}", ncols=100):
        sampled_data.append(entry)
    data = pd.DataFrame(sampled_data)
    data["category"] = category # add category column
    data.to_csv(cachepath, index=False)
    return data

def get_all_data(sample_size):
    # cache hit
    cachepath = dataset_path / f"cache_all_{sample_size}.csv"
    if cachepath.exists():
        data = pd.read_csv(cachepath)
        print(f"total data size: {data.memory_usage(deep=True).sum() / 1e9:.2f} gb")
        return data

    # cache miss
    data = pd.DataFrame()
    categories = get_all_categories()
    for category in tqdm(categories, desc="loading all data", ncols=100):
        category_data = get_category_data(category, sample_size)
        data = pd.concat([data, category_data], ignore_index=True)
        tqdm.write(f"loaded {category} - category size: {category_data.memory_usage(deep=True).sum() / 1e9:.2f} gb, total size: {data.memory_usage(deep=True).sum() / 1e9:.2f} gb")
    data.to_csv(cachepath, index=False)
    print(f"total data size: {data.memory_usage(deep=True).sum() / 1e9:.2f} gb")
    return data


# 
# preprocessing
# 



# - RQ1: Are reviews for some categories of product on Amazon overall more positive than for other categories? -> sentiment analysis: polarity
# - RQ2: Are reviews more subjective for some classes of products than for others? -> sentiment analysis: subjectivity
# - RQ3: Which aspects of different classes of products are the most important in the reviews? -> aspect-based sentiment analysis
# - RQ4: Can one predict the star rating from the review text? -> large language models or clustering embeddings

# 
# models
# 

# sentiment analysis (polarity)

# sentiment analysis (subjectivity)

# aspect-based sentiment analysis

# score estimation
# https://huggingface.co/LiYuan/amazon-review-sentiment-analysis





def preprocess(df):
    df = df.copy()

    # clean up
    df.drop_duplicates(inplace=True)
    df.drop(columns=['images', 'asin', 'parent_asin', 'user_id'], inplace=True, errors='ignore')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.dropna(subset=['text', 'title', 'rating'])
    df['text'] = df['text'].str.strip()
    df['title'] = df['title'].str.strip()

    # infer sentiment
    # ...
    return df



df = get_all_data(sample_size=10_000)
