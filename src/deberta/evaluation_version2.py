import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Config

from trainer import DebertaV3ForToxicClassification, ToxicDataset

# Paths
BASE_DIR = "/media/relive/37c3bf4d-5cdc-450a-a045-97d5f9bc78961/project_z"
OLD_RESULTS_DIR = os.path.join(BASE_DIR, "results", "bert")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "bert_new")


# Prefer GPU but fall back to CPU; warn if CUDA missing
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda")
else:
    device = torch.device("cpu")
    print("CUDA not available; using CPU (evaluation will be slower)")


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_path: str, model_name: str = "microsoft/deberta-v3-base"):
    config = DebertaV2Config.from_pretrained(model_name)
    config.num_labels = 2

    model = DebertaV3ForToxicClassification.from_pretrained(model_name, config=config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, test_dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\nEvaluating model...")

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def compute_metrics(all_labels, all_predictions, all_probabilities):
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")
    roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Macro:  {f1_macro:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=" * 50 + "\n")
    return accuracy, f1_macro, roc_auc


def plot_confusion_matrix(all_labels, all_predictions, output_dir: str):
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=["Non-Toxic", "Toxic"],
        yticklabels=["Non-Toxic", "Toxic"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_classification_report(all_labels, all_predictions, output_dir: str):
    report = classification_report(all_labels, all_predictions, output_dict=True, target_names=["Non-Toxic", "Toxic"])
    df_report = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:, :-1], annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Classification Report", fontsize=16)
    plt.tight_layout()
    path_img = os.path.join(output_dir, "classification_report.png")
    plt.savefig(path_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_img}")

    path_csv = os.path.join(output_dir, "classification_report.csv")
    df_report.to_csv(path_csv)
    print(f"Saved: {path_csv}")


def plot_roc_curve(all_labels, all_probabilities, output_dir: str):
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
    auc = roc_auc_score(all_labels, all_probabilities[:, 1])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    path = os.path.join(output_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_precision_recall(all_labels, all_probabilities, output_dir: str):
    precision, recall, _ = precision_recall_curve(all_labels, all_probabilities[:, 1])

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#3498db", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_metrics_bar(accuracy, f1_macro, roc_auc, output_dir: str):
    plt.figure(figsize=(7, 5))
    metrics = {"Accuracy": accuracy, "F1-Macro": f1_macro, "ROC-AUC": roc_auc}
    bars = plt.bar(metrics.keys(), metrics.values(), color=["#3c8dbc", "#e74c3c", "#2ecc71"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Overall Metrics")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha="center")
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_bar.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_classwise_scores(all_labels, all_predictions, output_dir: str):
    report = classification_report(all_labels, all_predictions, output_dict=True, target_names=["Non-Toxic", "Toxic"])
    df_report = pd.DataFrame(report).transpose()
    class_rows = df_report.loc[["Non-Toxic", "Toxic"], ["precision", "recall", "f1-score"]]

    ax = class_rows.plot(kind="bar", figsize=(8, 6), color=["#3498db", "#f39c12", "#2ecc71"])
    plt.ylim(0, 1)
    plt.title("Per-class Precision / Recall / F1")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "classwise_scores.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_breakdown(all_labels, all_predictions, output_dir: str):
    fp = ((all_labels == 0) & (all_predictions == 1)).sum()
    fn = ((all_labels == 1) & (all_predictions == 0)).sum()
    data = pd.DataFrame({"type": ["False Positives", "False Negatives"], "count": [fp, fn]})

    plt.figure(figsize=(6, 5))
    bars = plt.bar(data["type"], data["count"], color=["#e67e22", "#c0392b"])
    plt.ylabel("Count")
    plt.title("Error Breakdown")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, int(bar.get_height()),
                 ha="center", va="bottom")

    plt.tight_layout()
    path = os.path.join(output_dir, "error_breakdown.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_confidence(errors_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(8, 5))
    sns.histplot(errors_df["confidence"], bins=30, color="#9b59b6", kde=True, alpha=0.7)
    plt.xlabel("Model confidence (for predicted label)")
    plt.title("Error Confidence Distribution")
    plt.tight_layout()
    path = os.path.join(output_dir, "error_confidence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_predictions(all_labels, all_predictions, all_probabilities, texts, output_dir: str):
    df = pd.DataFrame(
        {
            "text": texts,
            "true_label": all_labels,
            "predicted_label": all_predictions,
            "prob_non_toxic": all_probabilities[:, 0],
            "prob_toxic": all_probabilities[:, 1],
        }
    )
    path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def save_errors(pred_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    errors_df = pred_df[pred_df["true_label"] != pred_df["predicted_label"]].copy()
    path = os.path.join(output_dir, "error_analysis.csv")
    errors_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return errors_df


def save_metrics(accuracy, f1_macro, roc_auc, output_dir: str):
    metrics = {"Accuracy": accuracy, "F1-Macro": f1_macro, "ROC-AUC": roc_auc}
    df = pd.DataFrame([metrics])
    path = os.path.join(output_dir, "test_metrics.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main():
    ensure_dirs()

    test_path = os.path.join(OLD_RESULTS_DIR, "test_data.csv")
    model_path = os.path.join(OLD_RESULTS_DIR, "deberta_toxic_model.pt")

    if not (os.path.exists(test_path) and os.path.exists(model_path)):
        print("Missing test data or model. Please ensure previous training artifacts exist.")
        sys.exit(1)

    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pin_memory = device.type == "cuda"
    test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=pin_memory)

    print("Loading trained model...")
    model = load_model(model_path, model_name)

    all_labels, all_predictions, all_probabilities = evaluate_model(model, test_dataloader)
    accuracy, f1_macro, roc_auc = compute_metrics(all_labels, all_predictions, all_probabilities)

    # Save raw predictions/errors first
    pred_df = save_predictions(all_labels, all_predictions, all_probabilities, test_texts, OUTPUT_DIR)
    errors_df = save_errors(pred_df, OUTPUT_DIR)

    # Plots
    plot_confusion_matrix(all_labels, all_predictions, OUTPUT_DIR)
    plot_classification_report(all_labels, all_predictions, OUTPUT_DIR)
    plot_roc_curve(all_labels, all_probabilities, OUTPUT_DIR)
    plot_precision_recall(all_labels, all_probabilities, OUTPUT_DIR)
    plot_metrics_bar(accuracy, f1_macro, roc_auc, OUTPUT_DIR)
    plot_classwise_scores(all_labels, all_predictions, OUTPUT_DIR)
    plot_error_breakdown(all_labels, all_predictions, OUTPUT_DIR)
    if not errors_df.empty:
        plot_error_confidence(errors_df, OUTPUT_DIR)

    # Metrics CSV
    save_metrics(accuracy, f1_macro, roc_auc, OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE (version 2)")
    print("=" * 50)
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
