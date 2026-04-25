import torch
from transformers import GPT2ForSequenceClassification, BertTokenizer
from peft import PeftModel
import os
import re

def preprocess_text(text):
    return text

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model and weight paths
    base_model_path = './models/gpt2-chinese'
    lora_model_path = './best_weibo_sentiment_lora'

    print("Loading model and tokenizer...")

    # Check if LoRA model exists
    if not os.path.exists(lora_model_path):
        print(f"Error: LoRA model path not found {lora_model_path}")
        print("Please run train.py first for training")
        return

    # Load tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '[PAD]'
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Please ensure models/gpt2-chinese directory contains tokenizer files")
        return

    # Load base model
    try:
        base_model = GPT2ForSequenceClassification.from_pretrained(
            base_model_path, 
            num_labels=2
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        print(f"Failed to load base model: {e}")
        print("Please ensure models/gpt2-chinese directory contains model files")
        return

    # Load LoRA weights
    try:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.to(device)
        model.eval()
        print("LoRA model loaded successfully!")
    except Exception as e:
        print(f"Failed to load LoRA weights: {e}")
        print("Please ensure LoRA weight files exist and are in correct format")
        return
    
    print("\n============= Weibo Sentiment Analysis (LoRA Version) =============")
    print("Enter Weibo content for analysis (input 'q' to quit):")
    
    while True:
        text = input("\nPlease enter Weibo content: ")
        if text.lower() == 'q':
            break
        
        if not text.strip():
            print("Input cannot be empty, please re-enter")
            continue
        
        try:
            # Preprocess text
            processed_text = preprocess_text(text)

            # Encode text
            encoding = tokenizer(
                processed_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Predict
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()

            # Output result
            confidence = probabilities[0][prediction].item()
            label = "Positive sentiment" if prediction == 1 else "Negative sentiment"
            
            print(f"Prediction result: {label} (confidence: {confidence:.4f})")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

if __name__ == "__main__":
    main()