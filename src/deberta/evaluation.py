import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Config
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from trainer import DebertaV3ForToxicClassification, ToxicDataset

# Prefer GPU but fall back to CPU; warn if CUDA missing
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device: cuda')
else:
    device = torch.device('cpu')
    print('CUDA not available; using CPU (evaluation will be slower)')


def load_model(model_path, model_name='microsoft/deberta-v3-base'):
    config = DebertaV2Config.from_pretrained(model_name)
    config.num_labels = 2
    
    model = DebertaV3ForToxicClassification.from_pretrained(
        model_name,
        config=config
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, test_dataloader):
    print("\nEvaluating model...")
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    return all_labels, all_predictions, all_probabilities


def compute_metrics(all_labels, all_predictions, all_probabilities):
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    if len(np.unique(all_labels)) == 2:
        roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
    
    print('\n' + '='*50)
    print('EVALUATION METRICS')
    print('='*50)
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'F1-Macro:  {f1_macro:.4f}')
    print(f'ROC-AUC:   {roc_auc:.4f}')
    print('='*50 + '\n')
    
    return accuracy, f1_macro, roc_auc


def plot_confusion_matrix(all_labels, all_predictions, output_dir):
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrix.png")


def plot_classification_report(all_labels, all_predictions, output_dir):
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    
    report = classification_report(all_labels, all_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:, :-1], annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Classification Report', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/classification_report.png")
    
    df_report.to_csv(f'{output_dir}/classification_report.csv')
    print(f"Saved: {output_dir}/classification_report.csv")


def error_analysis(all_labels, all_predictions, all_probabilities, test_texts, output_dir):
    errors = all_predictions != all_labels
    error_indices = np.where(errors)[0]
    
    error_data = []
    for idx in error_indices:
        error_data.append({
            'text': test_texts[idx],
            'true_label': all_labels[idx],
            'predicted_label': all_predictions[idx],
            'confidence': all_probabilities[idx][all_predictions[idx]],
            'true_prob': all_probabilities[idx][all_labels[idx]]
        })
    
    error_df = pd.DataFrame(error_data)
    
    print('\n' + '='*50)
    print('ERROR ANALYSIS')
    print('='*50)
    print(f'Total Errors: {len(error_indices)}')
    print(f'Error Rate: {len(error_indices) / len(all_labels) * 100:.2f}%')
    print('='*50 + '\n')
    
    if len(error_df) > 0:
        print('Sample Errors:')
        print(error_df.head(10))
        
        error_df.to_csv(f'{output_dir}/error_analysis.csv', index=False)
        print(f"\nSaved: {output_dir}/error_analysis.csv")


def save_metrics(accuracy, f1_macro, roc_auc, output_dir):
    metrics = {
        'Accuracy': accuracy,
        'F1-Macro': f1_macro,
        'ROC-AUC': roc_auc
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{output_dir}/test_metrics.csv', index=False)
    print(f"Saved: {output_dir}/test_metrics.csv")


def main():
    output_dir = "/media/relive/37c3bf4d-5cdc-450a-a045-97d5f9bc78961/project_z/results/bert"
    model_path = f'{output_dir}/deberta_toxic_model.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run trainer.py first to train the model.")
        return
    
    print("Loading test data...")
    test_df = pd.read_csv(f'{output_dir}/test_data.csv')
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    model_name = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pin_memory = device.type == 'cuda'
    test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=pin_memory)
    
    print("Loading trained model...")
    model = load_model(model_path, model_name)
    
    all_labels, all_predictions, all_probabilities = evaluate_model(model, test_dataloader)
    
    accuracy, f1_macro, roc_auc = compute_metrics(all_labels, all_predictions, all_probabilities)
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(all_labels, all_predictions, output_dir)
    plot_classification_report(all_labels, all_predictions, output_dir)
    
    print("\nPerforming error analysis...")
    error_analysis(all_labels, all_predictions, all_probabilities, test_texts, output_dir)
    
    print("\nSaving metrics...")
    save_metrics(accuracy, f1_macro, roc_auc, output_dir)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
