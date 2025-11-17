import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cross_validate_models(
    models: Dict, X, y: np.ndarray, k_folds: int, results_path: Optional[str] = None
) -> pd.DataFrame:
    cv_results = []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"\n{model_name} (CV {k_folds} folds)")
        cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
        cv_roc_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

        cv_results.append(
            {
                "Model": model_name,
                "CV_Accuracy": cv_accuracy.mean(),
                "CV_Accuracy_Std": cv_accuracy.std(),
                "CV_F1_Macro": cv_f1.mean(),
                "CV_F1_Macro_Std": cv_f1.std(),
                "CV_ROC_AUC": cv_roc_auc.mean(),
                "CV_ROC_AUC_Std": cv_roc_auc.std(),
            }
        )

        print(
            f"  Accuracy {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f}) | "
            f"F1 {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f}) | "
            f"ROC-AUC {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std():.4f})"
        )

    cv_df = pd.DataFrame(cv_results)
    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        cv_df.to_csv(results_path, index=False)
        print(f"Saved cross-validation results: {results_path}")
    return cv_df


def evaluate_models(models: Dict, X_train, y_train: np.ndarray, X_test, y_test: np.ndarray, results_path: str) -> Tuple[pd.DataFrame, Dict]:
    test_results = []
    predictions = {}

    for model_name, model in models.items():
        print(f"\n{model_name} (train + test)")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        test_results.append({"Model": model_name, "Accuracy": accuracy, "F1_Macro": f1, "ROC_AUC": roc_auc})
        predictions[model_name] = {"y_test": y_test, "y_pred": y_pred, "y_pred_proba": y_pred_proba}

        print(f"  Accuracy {accuracy:.4f} | F1 {f1:.4f} | ROC-AUC {roc_auc:.4f}")

    test_df = pd.DataFrame(test_results)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    test_df.to_csv(results_path, index=False)
    print(f"Saved test-set results: {results_path}")
    return test_df, predictions


def plot_model_comparison(cv_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    metrics = ["Accuracy", "F1_Macro", "ROC_AUC"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for idx, metric in enumerate(metrics):
        ax = axes[0, idx]
        cv_col = f"CV_{metric}"
        cv_std_col = f"CV_{metric}_Std"

        bars = ax.bar(cv_df["Model"], cv_df[cv_col], yerr=cv_df[cv_std_col], capsize=5, color=colors[idx], alpha=0.7)
        ax.set_ylabel("Score")
        ax.set_title(f"Cross-Validation {metric}")
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    for idx, metric in enumerate(metrics):
        ax = axes[1, idx]
        bars = ax.bar(test_df["Model"], test_df[metric], color=colors[idx], alpha=0.7)
        ax.set_ylabel("Score")
        ax.set_title(f"Test Set {metric}")
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot: {filepath}")
    return filepath


def plot_confusion_matrices(predictions: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(preds["y_test"], preds["y_pred"])
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=True,
            square=True,
            ax=axes[idx],
            xticklabels=["Non-Toxic", "Toxic"],
            yticklabels=["Non-Toxic", "Toxic"],
        )
        axes[idx].set_title(f"{model_name}\nConfusion Matrix", fontweight="bold")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices: {filepath}")
    return filepath


def plot_roc_curves(predictions: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (model_name, preds) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(preds["y_test"], preds["y_pred_proba"])
        auc = roc_auc_score(preds["y_test"], preds["y_pred_proba"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2, color=colors[idx % len(colors)])

    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves: {filepath}")
    return filepath


def plot_precision_recall(predictions: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (model_name, preds) in enumerate(predictions.items()):
        precision, recall, _ = precision_recall_curve(preds["y_test"], preds["y_pred_proba"])
        plt.plot(recall, precision, label=model_name, linewidth=2, color=colors[idx % len(colors)])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "precision_recall_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved precision-recall curves: {filepath}")
    return filepath


def error_analysis(predictions: Dict, test_texts: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for model_name, preds in predictions.items():
        y_test_curr = preds["y_test"]
        y_pred = preds["y_pred"]

        misclassified_mask = y_test_curr != y_pred
        fp_mask = (y_pred == 1) & (y_test_curr == 0)
        fn_mask = (y_pred == 0) & (y_test_curr == 1)

        results.append(
            {
                "Model": model_name,
                "Total_Errors": misclassified_mask.sum(),
                "False_Positives": fp_mask.sum(),
                "False_Negatives": fn_mask.sum(),
                "Error_Rate": misclassified_mask.mean(),
            }
        )

        if len(test_texts) == len(y_test_curr):
            fp_examples = test_texts[fp_mask].head(10)
            fn_examples = test_texts[fn_mask].head(10)
            fp_examples.to_csv(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_false_positives.csv"), index=False)
            fn_examples.to_csv(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_false_negatives.csv"), index=False)

    error_df = pd.DataFrame(results)
    error_df.to_csv(os.path.join(output_dir, "error_analysis_summary.csv"), index=False)
    print(f"Saved error analysis summary: {os.path.join(output_dir, 'error_analysis_summary.csv')}")
    return error_df


def plot_error_bars(error_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_names = error_df["Model"]
    x_pos = np.arange(len(model_names))
    width = 0.35

    axes[0].bar(x_pos - width / 2, error_df["False_Positives"], width, label="False Positives", color="orange", alpha=0.7)
    axes[0].bar(x_pos + width / 2, error_df["False_Negatives"], width, label="False Negatives", color="red", alpha=0.7)
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Type Distribution by Model")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(model_names, error_df["Error_Rate"], color="crimson", alpha=0.7)
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Error Rate")
    axes[1].set_title("Overall Error Rate by Model")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    for i, v in enumerate(error_df["Error_Rate"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "error_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved error analysis plots: {filepath}")
    return filepath


def export_classification_reports(predictions: Dict, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    report_paths = {}

    for model_name, preds in predictions.items():
        report = classification_report(preds["y_test"], preds["y_pred"], target_names=["Non-Toxic", "Toxic"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        filepath = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_classification_report.csv")
        report_df.to_csv(filepath)
        report_paths[model_name] = filepath
        print(f"Saved classification report for {model_name}: {filepath}")

    return report_paths
