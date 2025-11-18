import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import save_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

try:
    from .evaluation import (
        cross_validate_models,
        error_analysis,
        evaluate_models,
        export_classification_reports,
        plot_confusion_matrices,
        plot_error_bars,
        plot_model_comparison,
        plot_precision_recall,
        plot_roc_curves,
    )
    from .preprocessing import (
        build_sentence_embeddings,
        build_tfidf_features,
        detect_columns,
        load_raw_dataset,
        plot_class_distribution,
        plot_tfidf_top_features,
        plot_word_stats,
        preprocess_texts,
    )
except ImportError:
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from ml_based.evaluation import (
        cross_validate_models,
        error_analysis,
        evaluate_models,
        export_classification_reports,
        plot_confusion_matrices,
        plot_error_bars,
        plot_model_comparison,
        plot_precision_recall,
        plot_roc_curves,
    )
    from ml_based.preprocessing import (
        build_sentence_embeddings,
        build_tfidf_features,
        detect_columns,
        load_raw_dataset,
        plot_class_distribution,
        plot_tfidf_top_features,
        plot_word_stats,
        preprocess_texts,
    )

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def prepare_output_dirs(config: Dict) -> Dict[str, str]:
    base_results = config["output"]["results_dir"]
    tfidf_dir = os.path.join(base_results, "tfidf")
    embed_dir = os.path.join(base_results, "embeddings")

    tfidf_plots = os.path.join(tfidf_dir, config["output"].get("plots_subdir", "plots"))
    embed_plots = os.path.join(embed_dir, config["output"].get("plots_subdir", "plots"))

    for path in [base_results, tfidf_dir, embed_dir, tfidf_plots, embed_plots]:
        os.makedirs(path, exist_ok=True)

    return {
        "base": base_results,
        "tfidf": tfidf_dir,
        "embed": embed_dir,
        "tfidf_plots": tfidf_plots,
        "embed_plots": embed_plots,
    }


def save_splits(
    X_train, X_test, y_train, y_test, train_texts, test_texts, results_dir: str, prefix: str
) -> None:
    save_npz(os.path.join(results_dir, f"X_train_{prefix}.npz"), X_train)
    save_npz(os.path.join(results_dir, f"X_test_{prefix}.npz"), X_test)
    np.save(os.path.join(results_dir, f"y_train.npy"), y_train)
    np.save(os.path.join(results_dir, f"y_test.npy"), y_test)

    pd.DataFrame({"text": train_texts, "label": y_train}).to_csv(
        os.path.join(results_dir, "train_texts.csv"), index=False
    )
    pd.DataFrame({"text": test_texts, "label": y_test}).to_csv(
        os.path.join(results_dir, "test_texts.csv"), index=False
    )

    print(f"Saved datasets and splits to {results_dir}")


def build_models(config: Dict) -> Dict:
    models = {
        "Logistic Regression (TF-IDF)": LogisticRegression(
            max_iter=config["models"]["logistic_regression"]["max_iter"],
            random_state=config["dataset"]["random_state"],
            class_weight=config["models"]["logistic_regression"]["class_weight"],
            C=config["models"]["logistic_regression"]["C"],
        ),
        "Linear SVM (TF-IDF)": LinearSVC(
            max_iter=config["models"]["linear_svm"]["max_iter"],
            random_state=config["dataset"]["random_state"],
            class_weight=config["models"]["linear_svm"]["class_weight"],
            C=config["models"]["linear_svm"]["C"],
        ),
        "Naive Bayes (TF-IDF)": MultinomialNB(alpha=config["models"]["naive_bayes"]["alpha"]),
    }
    return models


def build_embedding_model(config: Dict) -> Dict:
    embed_cfg = config["models"]["embedding_log_regression"]
    return {
        "Logistic Regression (Embeddings)": LogisticRegression(
            max_iter=embed_cfg["max_iter"],
            class_weight=embed_cfg["class_weight"],
            C=embed_cfg["C"],
            random_state=config["dataset"]["random_state"],
        )
    }


def main(config_path: str = os.path.join(os.path.dirname(__file__), "config.yaml")) -> None:
    config = load_config(config_path)
    dirs = prepare_output_dirs(config)

    df = load_raw_dataset(config)
    text_col, label_col = detect_columns(df, config)
    processed_texts, labels = preprocess_texts(df, text_col, label_col, config)

    plot_class_distribution(labels, dirs["tfidf_plots"])
    plot_word_stats(processed_texts, labels, dirs["tfidf_plots"])

    vectorizer, tfidf_features = build_tfidf_features(processed_texts, config)
    plot_tfidf_top_features(vectorizer, tfidf_features, labels, dirs["tfidf_plots"])

    indices = np.arange(len(labels))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        tfidf_features,
        labels,
        indices,
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"],
        stratify=labels,
        shuffle=True,
    )

    train_texts = [processed_texts[i] for i in train_idx]
    test_texts = [processed_texts[i] for i in test_idx]
    save_splits(
        X_train, X_test, y_train, y_test, train_texts, test_texts, dirs["tfidf"], prefix="tfidf"
    )

    models = build_models(config)
    cv_df_tfidf = cross_validate_models(
        models,
        X_train,
        y_train,
        k_folds=config["dataset"]["k_folds"],
    )
    test_df_tfidf, predictions = evaluate_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        results_path=os.path.join(dirs["tfidf"], "test_results.csv"),
    )

    embed_predictions = {}
    cv_df_emb = pd.DataFrame()
    test_df_emb = pd.DataFrame()
    try:
        embeddings = build_sentence_embeddings(processed_texts, config)
        X_train_emb, X_test_emb = embeddings[train_idx], embeddings[test_idx]
        embed_models = build_embedding_model(config)
        cv_df_emb = cross_validate_models(
            embed_models, X_train_emb, y_train, k_folds=config["dataset"]["k_folds"]
        )
        cv_df_emb.to_csv(os.path.join(dirs["embed"], "cross_validation_results.csv"), index=False)
        test_df_emb, embed_predictions = evaluate_models(
            embed_models,
            X_train_emb,
            y_train,
            X_test_emb,
            y_test,
            results_path=os.path.join(dirs["embed"], "test_results.csv"),
        )
    except ImportError as exc:
        print(f"Embedding comparison skipped: {exc}")

    cv_df_tfidf.to_csv(os.path.join(dirs["tfidf"], "cross_validation_results.csv"), index=False)
    plot_model_comparison(cv_df_tfidf, test_df_tfidf, dirs["tfidf_plots"])
    plot_confusion_matrices(predictions, dirs["tfidf_plots"])
    plot_roc_curves(predictions, dirs["tfidf_plots"])
    plot_precision_recall(predictions, dirs["tfidf_plots"])

    error_df_tfidf = error_analysis(predictions, pd.DataFrame({"text": test_texts, "label": y_test}), dirs["tfidf_plots"])
    plot_error_bars(error_df_tfidf, dirs["tfidf_plots"])
    export_classification_reports(predictions, dirs["tfidf"])

    if not cv_df_emb.empty and embed_predictions:
        plot_model_comparison(cv_df_emb, test_df_emb, dirs["embed_plots"])
        plot_confusion_matrices(embed_predictions, dirs["embed_plots"])
        plot_roc_curves(embed_predictions, dirs["embed_plots"])
        plot_precision_recall(embed_predictions, dirs["embed_plots"])

        error_df_emb = error_analysis(
            embed_predictions, pd.DataFrame({"text": test_texts, "label": y_test}), dirs["embed_plots"]
        )
        plot_error_bars(error_df_emb, dirs["embed_plots"])
        export_classification_reports(embed_predictions, dirs["embed"])

    print(f"\nTF-IDF outputs saved under: {dirs['tfidf']}")
    if not cv_df_emb.empty:
        print(f"Embedding outputs saved under: {dirs['embed']}")
    else:
        print("Embedding outputs skipped (missing dependency).")


if __name__ == "__main__":
    main()