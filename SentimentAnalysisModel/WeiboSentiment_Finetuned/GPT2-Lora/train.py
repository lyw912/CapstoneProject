import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2ForSequenceClassification, 
    BertTokenizer, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Import LoRA components from PEFT library
from peft import LoraConfig, TaskType, get_peft_model

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define Weibo sentiment analysis dataset
class WeiboSentimentDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            review,
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

# Training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=3):
    best_f1 = 0.0

    for epoch in range(epochs):
        print(f"======== Epoch {epoch+1} / {epochs} ========")
        model.train()
        total_loss = 0

        # Training loop
        progress_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for batch in progress_bar:
            # Move data to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Parameter update
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluate model
        val_metrics = evaluate_model(model, val_dataloader, device)
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1 Score: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            # Save LoRA weights
            model.save_pretrained("./best_weibo_sentiment_lora")
            print("Saved best LoRA model!")

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    # Set local model save path
    model_name = 'uer/gpt2-chinese-cluecorpussmall'
    local_model_path = './models/gpt2-chinese'

    # Ensure directories exist
    os.makedirs(local_model_path, exist_ok=True)
    os.makedirs('./best_weibo_sentiment_lora', exist_ok=True)
    
    # Load dataset
    print("Loading Weibo sentiment dataset...")
    df = pd.read_csv('dataset/weibo_senti_100k.csv')

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # Load tokenizer
    print("Loading pre-trained model and tokenizer...")

    # Check if model exists locally
    if os.path.exists(os.path.join(local_model_path, 'config.json')):
        print(f"Loading tokenizer from local path: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
    else:
        print(f"Downloading tokenizer from Hugging Face to: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=local_model_path)
        # Save tokenizer to local
        tokenizer.save_pretrained(local_model_path)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
    
    # Record pad_token ID
    pad_token_id = tokenizer.pad_token_id

    # Create datasets
    train_dataset = WeiboSentimentDataset(
        train_df['review'].values,
        train_df['label'].values,
        tokenizer
    )
    
    val_dataset = WeiboSentimentDataset(
        val_df['review'].values,
        val_df['label'].values,
        tokenizer
    )
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained GPT2 model
    print("Loading GPT2 model...")
    if (os.path.exists(os.path.join(local_model_path, 'pytorch_model.bin')) or
        os.path.exists(os.path.join(local_model_path, 'model.safetensors'))):
        print(f"Loading model weights from local path: {local_model_path}")
        model = GPT2ForSequenceClassification.from_pretrained(local_model_path, num_labels=2)
    else:
        print(f"Downloading model weights from Hugging Face to: {local_model_path}")
        # Download full model directly from Hugging Face and save
        model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.save_pretrained(local_model_path)

    # Ensure model uses same pad_token_id as tokenizer
    model.config.pad_token_id = pad_token_id

    # Configure LoRA parameters
    print("Configuring LoRA parameters...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        target_modules=["c_attn", "c_proj"],  # GPT2 attention projection layers
        inference_mode=False,  # Training mode
        r=8,  # LoRA rank, controls number of trainable parameters
        lora_alpha=32,  # LoRA alpha parameter, scaling factor
        lora_dropout=0.1,  # LoRA Dropout
    )

    # Convert model to PEFT LoRA format
    print("Creating LoRA model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameter count and ratio

    model.to(device)

    # Set optimizer and learning rate scheduler
    print("Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),  # PEFT will handle parameter filtering automatically
        lr=5e-4,  # LoRA typically uses higher learning rate
        eps=1e-8
    )

    # Set total training steps and warmup steps
    total_steps = len(train_dataloader) * 3  # 3 epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=3
    )

    print("Training completed!")
    print("LoRA weights saved to: ./best_weibo_sentiment_lora/")

if __name__ == "__main__":
    main()