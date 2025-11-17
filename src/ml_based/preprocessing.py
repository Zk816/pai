import os
import re
import string
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Ensure required NLTK assets are available at runtime
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class TextCleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.stop_words = set(stopwords.words(config.get("stopwords_language", "english")))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        if self.config.get("lowercase", True):
            text = text.lower()

        if self.config.get("remove_urls", True):
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        if self.config.get("remove_mentions_hashtags", True):
            text = re.sub(r"@\w+|#\w+", "", text)

        if self.config.get("remove_punctuation", True):
            text = text.translate(str.maketrans("", "", string.punctuation))

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        if not self.config.get("remove_stopwords", True):
            return text
        words = text.split()
        return " ".join([word for word in words if word not in self.stop_words])

    def lemmatize(self, text: str) -> str:
        if not self.config.get("lemmatize", True):
            return text
        words = text.split()
        return " ".join([self.lemmatizer.lemmatize(word) for word in words])

    def __call__(self, text: str) -> str:
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text


def load_raw_dataset(config: Dict) -> pd.DataFrame:
    dataset_name = config["dataset"]["name"]
    split = config["dataset"].get("split", "train")
    print(f"Loading dataset {dataset_name!r} ({split})...")
    dataset = load_dataset(dataset_name, split=split)
    return pd.DataFrame(dataset)


def detect_columns(df: pd.DataFrame, config: Dict) -> Tuple[str, str]:
    text_col = config["dataset"].get("text_column")
    label_col = config["dataset"].get("label_column")

    if text_col is None:
        for col in df.columns:
            if "text" in col.lower() or "comment" in col.lower():
                text_col = col
                break
    if label_col is None:
        for col in df.columns:
            if "label" in col.lower() or "toxic" in col.lower():
                label_col = col
                break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer required columns. text_col={text_col}, label_col={label_col}"
        )

    print(f"Detected text column: {text_col}")
    print(f"Detected label column: {label_col}")
    return text_col, label_col


def preprocess_texts(df: pd.DataFrame, text_col: str, label_col: str, config: Dict) -> Tuple[List[str], np.ndarray]:
    cleaner = TextCleaner(config["preprocessing"])
    print("Cleaning text, removing stopwords, lemmatizing...")
    processed_texts = [cleaner(text) for text in df[text_col].astype(str)]
    labels = df[label_col].values

    # Collapse multilabel into binary if necessary
    if len(labels.shape) > 1:
        labels = (labels.sum(axis=1) > 0).astype(int)

    valid_indices = [idx for idx, text in enumerate(processed_texts) if text.strip()]
    processed_texts = [processed_texts[i] for i in valid_indices]
    labels = labels[valid_indices]

    print(f"Kept {len(processed_texts)} texts after cleaning")
    return processed_texts, labels


def plot_class_distribution(labels: np.ndarray, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(["Non-Toxic", "Toxic"], counts, color=["green", "red"], alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({height/sum(counts)*100:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution plot: {filepath}")
    return filepath


def plot_word_stats(texts: List[str], labels: np.ndarray, output_dir: str) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    toxic_texts = " ".join([text for text, label in zip(texts, labels) if label == 1])
    non_toxic_texts = " ".join([text for text, label in zip(texts, labels) if label == 0])

    toxic_words = Counter(toxic_texts.split())
    non_toxic_words = Counter(non_toxic_texts.split())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_n = 20

    toxic_common = toxic_words.most_common(top_n)
    words, counts = zip(*toxic_common) if toxic_common else ([], [])
    axes[0].barh(range(len(words)), counts, color="red", alpha=0.7)
    axes[0].set_yticks(range(len(words)))
    axes[0].set_yticklabels(words)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Frequency")
    axes[0].set_title(f"Top {top_n} Words in Toxic Comments")
    axes[0].grid(axis="x", alpha=0.3)

    non_toxic_common = non_toxic_words.most_common(top_n)
    words, counts = zip(*non_toxic_common) if non_toxic_common else ([], [])
    axes[1].barh(range(len(words)), counts, color="green", alpha=0.7)
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Frequency")
    axes[1].set_title(f"Top {top_n} Words in Non-Toxic Comments")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    barplot_path = os.path.join(output_dir, "word_frequencies_barplot.png")
    plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved word frequency barplot: {barplot_path}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    toxic_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Reds", max_words=100
    ).generate(toxic_texts)
    axes[0].imshow(toxic_wordcloud, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Word Cloud - Toxic Comments", fontsize=16, fontweight="bold")

    non_toxic_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Greens", max_words=100
    ).generate(non_toxic_texts)
    axes[1].imshow(non_toxic_wordcloud, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Word Cloud - Non-Toxic Comments", fontsize=16, fontweight="bold")

    plt.tight_layout()
    wordcloud_path = os.path.join(output_dir, "wordclouds.png")
    plt.savefig(wordcloud_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved word clouds: {wordcloud_path}")

    return barplot_path, wordcloud_path, toxic_wordcloud.words_


def build_tfidf_features(texts: List[str], config: Dict) -> Tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = TfidfVectorizer(
        max_features=config["features"]["tfidf"]["max_features"],
        ngram_range=tuple(config["features"]["tfidf"]["ngram_range"]),
        min_df=config["features"]["tfidf"]["min_df"],
        max_df=config["features"]["tfidf"]["max_df"],
    )
    features = vectorizer.fit_transform(texts)
    print(f"TF-IDF feature matrix shape: {features.shape}")
    return vectorizer, features


def plot_tfidf_top_features(vectorizer: TfidfVectorizer, features: csr_matrix, labels: np.ndarray, output_dir: str) -> str:
    feature_names = vectorizer.get_feature_names_out()
    toxic_mask = labels == 1
    non_toxic_mask = labels == 0

    toxic_mean = np.asarray(features[toxic_mask].mean(axis=0)).flatten()
    non_toxic_mean = np.asarray(features[non_toxic_mask].mean(axis=0)).flatten()

    top_n = 15
    toxic_top_indices = toxic_mean.argsort()[-top_n:][::-1]
    non_toxic_top_indices = non_toxic_mean.argsort()[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    toxic_features = [feature_names[i] for i in toxic_top_indices]
    toxic_scores = [toxic_mean[i] for i in toxic_top_indices]
    axes[0].barh(range(len(toxic_features)), toxic_scores, color="red", alpha=0.7)
    axes[0].set_yticks(range(len(toxic_features)))
    axes[0].set_yticklabels(toxic_features)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean TF-IDF Score")
    axes[0].set_title(f"Top {top_n} TF-IDF Features - Toxic Comments")
    axes[0].grid(axis="x", alpha=0.3)

    non_toxic_features = [feature_names[i] for i in non_toxic_top_indices]
    non_toxic_scores = [non_toxic_mean[i] for i in non_toxic_top_indices]
    axes[1].barh(range(len(non_toxic_features)), non_toxic_scores, color="green", alpha=0.7)
    axes[1].set_yticks(range(len(non_toxic_features)))
    axes[1].set_yticklabels(non_toxic_features)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean TF-IDF Score")
    axes[1].set_title(f"Top {top_n} TF-IDF Features - Non-Toxic Comments")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "tfidf_top_features.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved TF-IDF feature plot: {filepath}")
    return filepath


def build_sentence_embeddings(texts: List[str], config: Dict) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding comparison. "
            "Install with `pip install sentence-transformers`."
        ) from exc

    model_name = config["features"]["embedding"]["model_name"]
    batch_size = config["features"]["embedding"].get("batch_size", 32)
    print(f"Encoding texts with {model_name} (batch_size={batch_size})...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings
