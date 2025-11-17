import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|\#", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, text):
        words = text.split()
        return " ".join([w for w in words if w not in self.stop_words])

    def lemmatize_text(self, text):
        words = text.split()
        return " ".join([self.lemmatizer.lemmatize(w) for w in words])

    def full_preprocess(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text


def load_and_preprocess_data():
    print("Loading dataset...")
    dataset = load_dataset("AiresPucrs/toxic-comments", split="train")
    df = pd.DataFrame(dataset)
    
    print("Columns in dataset:", df.columns)
    
    text_col_candidates = [c for c in df.columns if "text" in c.lower()]
    if len(text_col_candidates) == 0:
        raise ValueError("No text column found!")
    text_col = text_col_candidates[0]
    
    print(f"\nDetected text column: {text_col}")
    
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()
    df["cleaned_text"] = df[text_col].apply(preprocessor.full_preprocess)
    
    print("\nText preprocessing complete!")
    print(df[[text_col, "cleaned_text"]].head())
    
    return df, text_col


def visualize_data(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating word frequency visualization...")
    all_words = []
    for text in df['cleaned_text']:
        all_words.extend(text.split())
    
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(20)
    words, counts = zip(*most_common)
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(words)), counts, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top 20 Most Frequent Words', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/word_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/word_frequency.png")
    
    print("\nGenerating word cloud...")
    text_for_cloud = ' '.join(df['cleaned_text'])
    
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text_for_cloud)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/wordcloud.png")


def split_data(df):
    print("\nSplitting data...")
    
    text_col = "cleaned_text"
    label_col = "toxic"
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )
    
    print(f"Train size: {len(train_texts)}")
    print(f"Val size: {len(val_texts)}")
    print(f"Test size: {len(test_texts)}")
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def save_splits(train_data, val_data, test_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data
    
    pd.DataFrame({'text': train_texts, 'label': train_labels}).to_csv(
        f'{output_dir}/train_data.csv', index=False
    )
    pd.DataFrame({'text': val_texts, 'label': val_labels}).to_csv(
        f'{output_dir}/val_data.csv', index=False
    )
    pd.DataFrame({'text': test_texts, 'label': test_labels}).to_csv(
        f'{output_dir}/test_data.csv', index=False
    )
    
    print(f"\nData splits saved to {output_dir}")


def main():
    output_dir = "/media/relive/37c3bf4d-5cdc-450a-a045-97d5f9bc78961/project_z/results/bert"
    
    df, text_col = load_and_preprocess_data()
    
    visualize_data(df, output_dir)
    
    train_data, val_data, test_data = split_data(df)
    
    save_splits(train_data, val_data, test_data, output_dir)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    
    return train_data, val_data, test_data, df


if __name__ == "__main__":
    main()