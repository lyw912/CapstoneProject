# -*- coding: utf-8 -*-
"""
Naive Bayes sentiment analysis model training script
"""
import argparse
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from base_model import BaseModel
from utils import stopwords


class BayesModel(BaseModel):
    """Naive Bayes sentiment analysis model"""
    
    def __init__(self):
        super().__init__("Bayes")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train Naive Bayes model

        Args:
            train_data: Training data in format [(text, label), ...]
            **kwargs: Other parameters
        """
        print(f"Starting {self.model_name} model training...")

        # Prepare data
        df_train = pd.DataFrame(train_data, columns=["words", "label"])

        # Feature encoding (bag-of-words model)
        print("Building bag-of-words model...")
        self.vectorizer = CountVectorizer(
            token_pattern=r'\[?\w+\]?', 
            stop_words=stopwords
        )
        
        X_train = self.vectorizer.fit_transform(df_train["words"])
        y_train = df_train["label"]

        print(f"Feature dimension: {X_train.shape[1]}")

        # Train model
        print("Training Naive Bayes classifier...")
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        print(f"{self.model_name} model training completed!")

    def predict(self, texts: List[str]) -> List[int]:
        """Predict text sentiment

        Args:
            texts: List of texts to predict

        Returns:
            List of predictions
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained, please call train method first")

        # Feature transformation
        X = self.vectorizer.transform(texts)

        # Predict
        predictions = self.model.predict(X)
        
        return predictions.tolist()
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment of a single text

        Args:
            text: Text to predict

        Returns:
            (predicted_label, confidence)
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained, please call train method first")

        # Feature transformation
        X = self.vectorizer.transform([text])

        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Naive Bayes sentiment analysis model training')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='Test data path')
    parser.add_argument('--model_path', type=str, default='./model/bayes_model.pkl',
                        help='Model save path')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing model, no training')
    
    args = parser.parse_args()
    
    # Create model
    model = BayesModel()
    
    if args.eval_only:
        # Evaluation only mode
        print("Evaluation mode: Loading existing model for evaluation")
        model.load_model(args.model_path)

        # Load test data
        _, test_data = BaseModel.load_data(args.train_path, args.test_path)

        # Evaluate model
        model.evaluate(test_data)
    else:
        # Training mode
        # Load data
        train_data, test_data = BaseModel.load_data(args.train_path, args.test_path)

        # Train model
        model.train(train_data)

        # Evaluate model
        model.evaluate(test_data)

        # Save model
        model.save_model(args.model_path)

        # Example predictions
        print("\nExample predictions:")
        test_texts = [
            "今天天气真好，心情很棒",
            "这部电影太无聊了，浪费时间",
            "哈哈哈，太有趣了"
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {text}")
            print(f"Prediction: {sentiment} (Confidence: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()