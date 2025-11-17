import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from preprocessing import load_and_preprocess_data, split_data

# Prefer GPU but fall back to CPU; warn if CUDA missing
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device: cuda')
else:
    device = torch.device('cpu')
    print('CUDA not available; using CPU (expect slower training)')


class DebertaV3ForToxicClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(train_data, val_data, test_data, tokenizer):
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data
    
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)
    test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)
    
    pin_memory = device.type == 'cuda'
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=pin_memory)
    
    print('Datasets created')
    
    return train_dataloader, val_dataloader, test_dataloader


def train_model(model, train_dataloader, val_dataloader, epochs, learning_rate, output_dir):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 50)
        
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc='Training', leave=True)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            preds = torch.argmax(logits, dim=1)
            batch_acc = (preds == labels).float().mean().item()
            
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc:.4f}"
            })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        print(f"Epoch Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved training curves to {output_dir}/training_curves.png")
    
    return model, train_losses, val_losses


def main():
    output_dir = "/media/relive/37c3bf4d-5cdc-450a-a045-97d5f9bc78961/project_z/results/bert"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(f'{output_dir}/train_data.csv'):
        print("Running preprocessing...")
        df, text_col = load_and_preprocess_data()
        train_data, val_data, test_data = split_data(df)
        
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
    else:
        print("Loading preprocessed data...")
        train_df = pd.read_csv(f'{output_dir}/train_data.csv')
        val_df = pd.read_csv(f'{output_dir}/val_data.csv')
        test_df = pd.read_csv(f'{output_dir}/test_data.csv')
        
        train_data = (train_df['text'].tolist(), train_df['label'].tolist())
        val_data = (val_df['text'].tolist(), val_df['label'].tolist())
        test_data = (test_df['text'].tolist(), test_df['label'].tolist())
    
    model_name = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_data, val_data, test_data, tokenizer
    )
    
    print("\nInitializing model...")
    label_col = "toxic"
    
    config = DebertaV2Config.from_pretrained(model_name)
    config.num_labels = 2
    
    model = DebertaV3ForToxicClassification.from_pretrained(
        model_name,
        config=config
    ).to(device)
    
    print("Model loaded successfully")
    
    epochs = 3
    learning_rate = 2e-5
    
    print("\nStarting training...")
    model, train_losses, val_losses = train_model(
        model, train_dataloader, val_dataloader, epochs, learning_rate, output_dir
    )
    
    model_path = f'{output_dir}/deberta_toxic_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved to {model_path}')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
