import json
import os

import pandas as pd
from tqdm import tqdm

from utils import *

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
    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(repo_id="McAuley-Lab/Amazon-Reviews-2023", filename="asin2category.json", repo_type="dataset", cache_dir=dataset_path)
    file = open(file_path, "r")
    data = json.load(file)
    file.close()
    return data


def get_category_metadata(category):
    import datasets

    data = datasets.load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", cache_dir=dataset_path, trust_remote_code=True)
    return data


def get_all_categories():
    import datasets

    data = datasets.load_dataset("text", data_files="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/all_categories.txt", streaming=False, cache_dir=dataset_path, trust_remote_code=True)
    return data["train"].to_dict()["text"]


def get_category_data(category, sample_size):
    import datasets

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
    data["category"] = category  # add category column
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


def get_language(review):
    from langdetect import detect

    try:
        return detect(review)
    except:
        return None


def get_sentiment(review):
    from transformers import pipeline

    model = pipeline("sentiment-analysis", device=get_device(disable_mps=False), model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", model_kwargs={"cache_dir": weights_path})
    review = review[:512]
    try:
        return model(review)[0]["label"], model(review)[0]["score"]
    except:
        return None, None


def get_subjectivity(review):
    from transformers import pipeline

    model = pipeline(task="text-classification", model="cffl/bert-base-styleclassification-subjective-neutral", top_k=None, device=get_device(disable_mps=False), model_kwargs={"cache_dir": weights_path})
    review = review[:512]
    try:
        outputs = model(review)[0]
        subjectivity_score = outputs[0]["score"]
        objectivity_score = outputs[1]["score"]
        return subjectivity_score
    except:
        return None


# def get_aspects(review):
#     from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

#     model_name = "yangheng/deberta-v3-large-absa-v1.1"
#     tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512, cache_dir=weights_path, local_files_only=False)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=weights_path, local_files_only=False)
#     classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device_map="auto")
#     review = review[:512]

#     outputs = classifier(review)
#     return outputs



def get_rating(review):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis", device=get_device(disable_mps=False), cache_dir=weights_path)
    model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    review = review[:512]
    try:
        inputs = tokenizer(review, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax().item()
        predicted_class = model.config.id2label[predicted_class_idx]
        return float(predicted_class[0])  # match dataset
    except:
        return None


def preprocess(df):
    df = df.copy()

    df.drop(columns=["images", "asin", "parent_asin", "user_id"], inplace=True, errors="ignore")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df.dropna(subset=["text", "title", "rating"])
    df["text"] = df["text"].str.replace(r"<.*?>", "", regex=True)  # drop html tags
    df["title"] = df["title"].str.replace(r"<.*?>", "", regex=True)
    df["text"] = df["text"].str.strip()
    df["title"] = df["title"].str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["title"].str.len() > 0]

    # make sure to store everything after you're done -> then we only load data from huggingface instead of doing all of this
    # for idx, row in tqdm(df.iterrows(), total=len(df), desc="sentiment analysis", ncols=100):
    #     review = f"{row['title']}: {row['text']}"
    #     language = get_language(review)
    #     df.at[idx, 'language'] = language
    return df


df = get_all_data(sample_size=100)
df = preprocess(df)

fst = df.iloc[11]
review = f"{fst['title']}: {fst['text']}"
print(review[:512])

# print(f"{get_language(review)=}")
# print(f"{get_sentiment(review)=}")
# print(f"{get_subjectivity(review)=}")
# print(f"{get_rating(review)=}")

from setfit import AbsaModel

model = AbsaModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    spacy_model="en_core_web_sm",
    device=get_device(disable_mps=False),
    cache_dir=weights_path,
)

