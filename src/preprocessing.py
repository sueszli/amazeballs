import json
import os

import pandas as pd
from tqdm import tqdm

from utils import *

set_env()

data_path = get_current_dir().parent / "data"
dataset_path = get_current_dir().parent / "datasets"
weights_path = get_current_dir().parent / "weights"

os.makedirs(data_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


#
# data loading
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
    return df


#
# inference
#


def get_language(review):
    from langdetect import detect

    try:
        return detect(review)
    except Exception as e:
        print("error get_language:", e)
        return None


def get_sentiment(review):
    from transformers import pipeline

    model = pipeline("sentiment-analysis", device=get_device(disable_mps=False), model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", model_kwargs={"cache_dir": weights_path})
    review = review[:512]
    try:
        return model(review)[0]["label"], model(review)[0]["score"]
    except Exception as e:
        print("error get_sentiment:", e)
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
    except Exception as e:
        print("error get_subjectivity:", e)
        return None


def get_aspects_old(review):
    # this takes >5s and SetFitABSA is not mature enough yet
    raise DeprecationWarning
    from pyabsa import AspectTermExtraction as ATEPC

    aspect_extractor = ATEPC.AspectExtractor("multilingual", auto_device=True, cal_perplexity=True, checkpoint_save_path=weights_path)
    review = review[:512]
    try:
        result = aspect_extractor.predict(review, pred_sentiment=True, print_result=False)
        aspects = result["aspect"]
        sentiments = result["sentiment"]
        confidences = result["confidence"]
        zipped = list(zip(aspects, sentiments, confidences))
        zipped = [(aspect, sentiment, confidence) for aspect, sentiment, confidence in zipped if confidence > 0.5]
        return zipped
    except:
        return None


def get_aspects(review):
    import yake
    from summa import keywords

    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.3, top=10, features=None)
    review = review[:512]
    try:
        keywords = kw_extractor.extract_keywords(review)
        return [kw for kw, score in keywords if score > 0.3]
    except Exception as e:
        print("error get_aspects:", e)
        return None


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
    except Exception as e:
        print("error get_rating:", e)
        return None


def add_inferences(df, sample_size):
    results_path = dataset_path / f"results_n{sample_size}.csv"
    print(f"results_path: {results_path}")
    if not results_path.exists():
        tqdm.pandas()

        def process_row(row):
            review = f"{row['title']}: {row['text']}"
            sentiment, score = get_sentiment(review)
            return {
                "language": get_language(review),
                "sentiment": sentiment,
                "sentiment_score": score,
                "subjectivity_score": get_subjectivity(review),
                "aspects": get_aspects(review),
                "rating": get_rating(review),
            }

        results = df.progress_apply(process_row, axis=1)
        results.to_csv(results_path, index=False)
    else:
        results = pd.read_csv(results_path)

    df = df.copy()
    results = results["0"].apply(lambda x: pd.Series(eval(x)))
    df = pd.concat([df, results], axis=1)
    return df


if __name__ == "__main__":
    df = get_all_data(sample_size=100)
    df = preprocess(df)
    df = add_inferences(df, sample_size=100)

    # save
    df.to_csv(data_path / "data.csv", index=False)
    print("saved data")