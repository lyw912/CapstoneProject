# -*- coding: utf-8 -*-
"""
Unified Sentiment Analysis Prediction Program
Supports loading all models for sentiment prediction
"""
import argparse
import os
import re
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# Import all model classes
from bayes_train import BayesModel
from svm_train import SVMModel
from xgboost_train import XGBoostModel
from lstm_train import LSTMModel
from bert_train import BertModel_Custom
from utils import processing


class SentimentPredictor:
    """Sentiment Analysis Predictor"""
    
    def __init__(self):
        self.models = {}
        self.available_models = {
            'bayes': BayesModel,
            'svm': SVMModel,
            'xgboost': XGBoostModel,
            'lstm': LSTMModel,
            'bert': BertModel_Custom
        }
        
    def load_model(self, model_type: str, model_path: str, **kwargs) -> None:
        """Load model of specified type
        
        Args:
            model_type: Model type ('bayes', 'svm', 'xgboost', 'lstm', 'bert')
            model_path: Model file path
            **kwargs: Other parameters (e.g., BERT pretrained model path)
        """
        if model_type not in self.available_models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file does not exist: {model_path}")
            return
        
        print(f"Loading {model_type.upper()} model...")
        
        try:
            if model_type == 'bert':
                # BERT requires additional pretrained model path
                bert_path = kwargs.get('bert_path', './model/chinese_wwm_pytorch')
                model = BertModel_Custom(bert_path)
            else:
                model = self.available_models[model_type]()
            
            model.load_model(model_path)
            self.models[model_type] = model
            print(f"{model_type.upper()} model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load {model_type.upper()} model: {e}")
    
    def load_all_models(self, model_dir: str = './model', bert_path: str = './model/chinese_wwm_pytorch') -> None:
        """Load all available models
        
        Args:
            model_dir: Model file directory
            bert_path: BERT pretrained model path
        """
        model_files = {
            'bayes': os.path.join(model_dir, 'bayes_model.pkl'),
            'svm': os.path.join(model_dir, 'svm_model.pkl'),
            'xgboost': os.path.join(model_dir, 'xgboost_model.pkl'),
            'lstm': os.path.join(model_dir, 'lstm_model.pth'),
            'bert': os.path.join(model_dir, 'bert_model.pth')
        }
        
        print("Starting to load all available models...")
        for model_type, model_path in model_files.items():
            self.load_model(model_type, model_path, bert_path=bert_path)
        
        print(f"\nLoaded {len(self.models)} models: {list(self.models.keys())}")
    
    def predict_single(self, text: str, model_type: str = None) -> Dict[str, Tuple[int, float]]:
        """Predict sentiment of a single text
        
        Args:
            text: Text to predict
            model_type: Specify model type, if None use all loaded models
            
        Returns:
            Dict[model_type, (prediction, confidence)]
        """
        # Text preprocessing
        processed_text = processing(text)
        
        if model_type:
            if model_type not in self.models:
                raise ValueError(f"Model {model_type} is not loaded")
            
            prediction, confidence = self.models[model_type].predict_single(processed_text)
            return {model_type: (prediction, confidence)}
        
        # Use all models for prediction
        results = {}
        for name, model in self.models.items():
            try:
                prediction, confidence = model.predict_single(processed_text)
                results[name] = (prediction, confidence)
            except Exception as e:
                print(f"Model {name} prediction failed: {e}")
                results[name] = (0, 0.0)
        
        return results
    
    def predict_batch(self, texts: List[str], model_type: str = None) -> Dict[str, List[int]]:
        """Batch predict text sentiment
        
        Args:
            texts: List of texts to predict
            model_type: Specify model type, if None use all loaded models
            
        Returns:
            Dict[model_type, predictions]
        """
        # Text preprocessing
        processed_texts = [processing(text) for text in texts]
        
        if model_type:
            if model_type not in self.models:
                raise ValueError(f"Model {model_type} is not loaded")
            
            predictions = self.models[model_type].predict(processed_texts)
            return {model_type: predictions}
        
        # Use all models for prediction
        results = {}
        for name, model in self.models.items():
            try:
                predictions = model.predict(processed_texts)
                results[name] = predictions
            except Exception as e:
                print(f"Model {name} prediction failed: {e}")
                results[name] = [0] * len(texts)
        
        return results
    
    def ensemble_predict(self, text: str, weights: Dict[str, float] = None) -> Tuple[int, float]:
        """Ensemble prediction (voting from multiple models)
        
        Args:
            text: Text to predict
            weights: Model weights, if None use equal weights
            
        Returns:
            (prediction, confidence)
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded")
        
        results = self.predict_single(text)
        
        if weights is None:
            weights = {name: 1.0 for name in results.keys()}
        
        # Weighted average
        total_weight = 0
        weighted_prob = 0
        
        for model_name, (pred, conf) in results.items():
            if model_name in weights:
                weight = weights[model_name]
                prob = conf if pred == 1 else 1 - conf
                weighted_prob += prob * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0, 0.5
        
        final_prob = weighted_prob / total_weight
        final_pred = int(final_prob > 0.5)
        final_conf = final_prob if final_pred == 1 else 1 - final_prob
        
        return final_pred, final_conf
    
    def interactive_predict(self):
        """Interactive prediction mode"""
        if len(self.models) == 0:
            print("Error: No models loaded, please load models first")
            return
        
        print("\n" + "="*50)
        print("="*50)
        print(f"Loaded models: {', '.join(self.models.keys())}")
        print("Enter 'q' to exit")
        print("Enter 'models' to view model list")
        print("Enter 'ensemble' to use ensemble prediction")
        print("-"*50)
        
        while True:
            try:
                text = input("\nPlease enter Weibo content to analyze: ").strip()
                
                if text.lower() == 'q':
                    print("👋 Goodbye!")
                    break
                
                if text.lower() == 'models':
                    print(f"Loaded models: {list(self.models.keys())}")
                    continue
                
                if text.lower() == 'ensemble':
                    if len(self.models) > 1:
                        pred, conf = self.ensemble_predict(text)
                        sentiment = "😊 Positive" if pred == 1 else "😞 Negative"
                        print(f"\n🤖 Ensemble prediction result:")
                        print(f"   Sentiment: {sentiment}")
                        print(f"   Confidence: {conf:.4f}")
                    else:
                        print("❌ Ensemble prediction requires at least 2 models")
                    continue
                
                if not text:
                    print("❌ Please enter valid content")
                    continue
                
                # Predict
                results = self.predict_single(text)
                
                print(f"\n📝 Original: {text}")
                print("🔍 Prediction results:")
                
                for model_name, (pred, conf) in results.items():
                    sentiment = "😊 Positive" if pred == 1 else "😞 Negative"
                    print(f"   {model_name.upper():8}: {sentiment} (Confidence: {conf:.4f})")
                
                # If multiple models, show ensemble result
                if len(results) > 1:
                    ensemble_pred, ensemble_conf = self.ensemble_predict(text)
                    ensemble_sentiment = "😊 Positive" if ensemble_pred == 1 else "😞 Negative"
                    print(f"   {'Ensemble':8}: {ensemble_sentiment} (Confidence: {ensemble_conf:.4f})")
                
            except KeyboardInterrupt:
                print("\n\n👋 Program interrupted, goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during prediction: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified Weibo Sentiment Analysis Prediction Program')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='Model file directory')
    parser.add_argument('--bert_path', type=str, default='./model/chinese_wwm_pytorch',
                        help='BERT pre-trained model path')
    parser.add_argument('--model_type', type=str, choices=['bayes', 'svm', 'xgboost', 'lstm', 'bert'],
                        help='指定单个模型类型进行预测')
    parser.add_argument('--text', type=str,
                        help='直接预测指定文本')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='Interactive prediction mode (default)')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用集成预测')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = SentimentPredictor()
    
    # Load models
    if args.model_type:
        # Load specified model
        model_files = {
            'bayes': 'bayes_model.pkl',
            'svm': 'svm_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lstm': 'lstm_model.pth',
            'bert': 'bert_model.pth'
        }
        model_path = os.path.join(args.model_dir, model_files[args.model_type])
        predictor.load_model(args.model_type, model_path, bert_path=args.bert_path)
    else:
        # Load all models
        predictor.load_all_models(args.model_dir, args.bert_path)
    
    # If text is specified, predict directly
    if args.text:
        if args.ensemble and len(predictor.models) > 1:
            pred, conf = predictor.ensemble_predict(args.text)
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {args.text}")
            print(f"Ensemble prediction: {sentiment} (Confidence: {conf:.4f})")
        else:
            results = predictor.predict_single(args.text, args.model_type)
            print(f"Text: {args.text}")
            for model_name, (pred, conf) in results.items():
                sentiment = "Positive" if pred == 1 else "Negative"
                print(f"{model_name.upper()}: {sentiment} (Confidence: {conf:.4f})")
    elif args.interactive:
        # Interactive mode
        predictor.interactive_predict()


if __name__ == "__main__":
    main()