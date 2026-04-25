# -*- coding: utf-8 -*-
"""
Qwen3 Model Base Class, Unified Interface
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


class BaseQwenModel(ABC):
    """Qwen3 Sentiment Analysis Model Base Class"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """Predict text sentiment"""
        pass
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment of a single text

        Args:
            text: Text to predict

        Returns:
            (predicted_label, confidence)
        """
        predictions = self.predict([text])
        return predictions[0], 0.0  # Default confidence is 0
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained yet, please call train method first")
            
        texts = [item[0] for item in test_data]
        labels = [item[1] for item in test_data]
        
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')

        print(f"\n{self.model_name} Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nDetailed Report:")
        print(classification_report(labels, predictions))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(labels, predictions)
        }
    
    @abstractmethod
    def save_model(self, model_path: str = None) -> None:
        """Save model to file"""
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load model from file"""
        pass
    
    @staticmethod
    def load_data(train_path: str = None, test_path: str = None, csv_path: str = 'dataset/weibo_senti_100k.csv') -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Load training and test data

        Args:
            train_path: Training data txt file path (optional)
            test_path: Test data txt file path (optional)
            csv_path: CSV data file path (default)
        """
        
        # Priority: try CSV file first
        if os.path.exists(csv_path):
            print(f"Loading data from CSV file: {csv_path}")
            df = pd.read_csv(csv_path)

            # Check data format
            if 'review' in df.columns and 'label' in df.columns:
                # Convert DataFrame to list of tuples
                data = [(row['review'], row['label']) for _, row in df.iterrows()]

                # Split training and test data, fixed test set of 5000 samples
                total_samples = len(data)
                if total_samples > 5000:
                    test_size = 5000
                    train_data, test_data = train_test_split(
                        data,
                        test_size=test_size,
                        random_state=42,
                        stratify=[label for _, label in data]
                    )
                else:
                    # If total data is less than 5000, use 20% as test set
                    train_data, test_data = train_test_split(
                        data,
                        test_size=0.2,
                        random_state=42,
                        stratify=[label for _, label in data]
                    )

                print(f"Training data size: {len(train_data)}")
                print(f"Test data size: {len(test_data)}")
                
                return train_data, test_data
            else:
                print(f"CSV file format is incorrect, missing 'review' or 'label' column")

        # If CSV does not exist, try txt files
        elif train_path and test_path and os.path.exists(train_path) and os.path.exists(test_path):
            def load_corpus(path):
                data = []
                with open(path, "r", encoding="utf8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            content = parts[0]
                            sentiment = int(parts[1])
                            data.append((content, sentiment))
                return data
            
            print("Loading training data from txt file...")
            train_data = load_corpus(train_path)
            print(f"Training data size: {len(train_data)}")

            print("Loading test data from txt file...")
            test_data = load_corpus(test_path)
            print(f"Test data size: {len(test_data)}")
            
            return train_data, test_data
        
        else:
            # If neither exists, provide guidance for creating sample data
            print("Data file not found!")
            print("Please ensure one of the following files exists:")
            print(f"1. CSV file: {csv_path}")
            print(f"2. txt files: {train_path} and {test_path}")
            print("\nData format requirements:")
            print("CSV file: contains 'review' and 'label' columns")
            print("txt file: each line in format 'text content\\tlabel'")
            
            # Create sample data
            sample_data = [
                ("今天天气真好，心情很棒!", 1),
                ("这部电影太无聊了", 0),
                ("非常喜欢这个产品", 1),
                ("服务态度很差", 0),
                ("质量不错，值得推荐", 1)
            ]

            print("Using sample data for demonstration...")
            train_data = sample_data * 20  # Expand sample data
            test_data = sample_data * 5

            return train_data, test_data