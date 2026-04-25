# -*- coding: utf-8 -*-
"""
SVM Sentiment Analysis Model Training Script
"""
import argparse
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

from base_model import BaseModel
from utils import stopwords


class SVMModel(BaseModel):
    """SVM Sentiment Analysis Model"""
    
    def __init__(self):
        super().__init__("SVM")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train SVM model
        
        Args:
            train_data: Training data, format as [(text, label), ...]
            **kwargs: Other parameters, supports kernel, C and other SVM parameters
        """
        print(f"Starting {self.model_name} model training...")
        
        # Prepare data
        df_train = pd.DataFrame(train_data, columns=["words", "label"])
        
        # Feature encoding (TF-IDF model)
        print("Building TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'\[?\w+\]?', 
            stop_words=stopwords
        )
        
        X_train = self.vectorizer.fit_transform(df_train["words"])
        y_train = df_train["label"]
        
        print(f"Feature dimension: {X_train.shape[1]}")
        
        # Get SVM parameters
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 1.0)
        gamma = kwargs.get('gamma', 'scale')
        
        # Train model
        print(f"Training SVM classifier (kernel={kernel}, C={C}, gamma={gamma})...")
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        print(f"{self.model_name} model training completed!")
        
    def predict(self, texts: List[str]) -> List[int]:
        """Predict text sentiment
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of prediction results
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained yet, please call train method first")
            
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
            raise ValueError(f"Model {self.model_name} is not trained yet, please call train method first")
            
        # Feature transformation
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SVM Sentiment Analysis Model Training')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='Test data path')
    parser.add_argument('--model_path', type=str, default='./model/svm_model.pkl',
                        help='Model save path')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help='SVM kernel type')
    parser.add_argument('--C', type=float, default=1.0,
                        help='SVM regularization parameter C')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='SVM kernel parameter gamma')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing model, no training')
    
    args = parser.parse_args()
    
    # Create model
    model = SVMModel()
    
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
        model.train(train_data, kernel=args.kernel, C=args.C, gamma=args.gamma)
        
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