#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 Weibo Sentiment Analysis Unified Prediction Interface
Supports 0.6B, 4B, 8B sizes for both Embedding and LoRA models
"""

import os
import sys
import argparse
import torch
from typing import List, Dict, Tuple, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_config import QWEN3_MODELS, MODEL_PATHS
from qwen3_embedding_universal import Qwen3EmbeddingUniversal
from qwen3_lora_universal import Qwen3LoRAUniversal


class Qwen3UniversalPredictor:
    """Qwen3 Unified Predictor"""

    def __init__(self):
        self.models = {}  # Stores loaded models {model_key: {model: obj, display_name: str}}
        
    def _get_model_key(self, model_type: str, model_size: str) -> str:
        """Generate model key"""
        return f"{model_type}_{model_size}"

    def load_model(self, model_type: str, model_size: str) -> None:
        """Load specified model"""
        if model_type not in ['embedding', 'lora']:
            raise ValueError(f"Unsupported model type: {model_type}")
        if model_size not in ['0.6B', '4B', '8B']:
            raise ValueError(f"Unsupported model size: {model_size}")
            
        model_path = MODEL_PATHS[model_type][model_size]
        model_key = self._get_model_key(model_type, model_size)
        
        # Check if trained model file exists
        if not os.path.exists(model_path):
            print(f"Trained model file does not exist: {model_path}")
            print(f"Please train the {model_type.upper()}-{model_size} model first, or check model path configuration")
            return

        print(f"Loading {model_type.upper()}-{model_size} model...")
        
        try:
            if model_type == 'embedding':
                model = Qwen3EmbeddingUniversal(model_size)
                model.load_model(model_path)
            else:  # lora
                model = Qwen3LoRAUniversal(model_size)
                model.load_model(model_path)
            
            self.models[model_key] = {
                'model': model,
                'display_name': f"Qwen3-{model_type.title()}-{model_size}"
            }
            print(f"{model_type.upper()}-{model_size} model loaded successfully")

        except Exception as e:
            print(f"Failed to load {model_type.upper()}-{model_size} model: {e}")
            print(f"This may be due to base model download failure or corrupted trained model file")
    
    def load_all_models(self, model_dir: str = './models') -> None:
        """Load all available models"""
        print("Starting to load all available Qwen3 models...")

        loaded_count = 0
        for model_type in ['embedding', 'lora']:
            for model_size in ['0.6B', '4B', '8B']:
                try:
                    self.load_model(model_type, model_size)
                    loaded_count += 1
                except Exception as e:
                    print(f"Skipping {model_type}-{model_size}: {e}")

        print(f"\nLoaded {loaded_count} models")
        self._print_loaded_models()
    
    def load_specific_models(self, model_configs: List[Tuple[str, str]]) -> None:
        """Load specified model configurations
        Args:
            model_configs: List of [(model_type, model_size), ...]
        """
        print("Loading specified Qwen3 models...")
        
        for model_type, model_size in model_configs:
            try:
                self.load_model(model_type, model_size)
            except Exception as e:
                print(f"Skipping {model_type}-{model_size}: {e}")

        print(f"\nLoaded {len(self.models)} models")
        self._print_loaded_models()

    def _print_loaded_models(self):
        """Print list of loaded models"""
        if self.models:
            print("Loaded models:")
            for model_info in self.models.values():
                print(f"  - {model_info['display_name']}")
        else:
            print("No models loaded successfully")
    
    def predict_single(self, text: str, model_key: str = None) -> Dict[str, Tuple[int, float]]:
        """Single text prediction
        Args:
            text: Text to predict
            model_key: Specified model key, None means use all models
        Returns:
            {model_name: (prediction, confidence), ...}
        """
        results = {}

        if model_key and model_key in self.models:
            # Use specified model
            model_info = self.models[model_key]
            try:
                prediction, confidence = model_info['model'].predict_single(text)
                results[model_info['display_name']] = (prediction, confidence)
            except Exception as e:
                print(f"Model {model_info['display_name']} prediction failed: {e}")
                results[model_info['display_name']] = (0, 0.0)
        else:
            # Use all models
            for model_info in self.models.values():
                try:
                    prediction, confidence = model_info['model'].predict_single(text)
                    results[model_info['display_name']] = (prediction, confidence)
                except Exception as e:
                    print(f"Model {model_info['display_name']} prediction failed: {e}")
                    results[model_info['display_name']] = (0, 0.0)
        
        return results
    
    def predict_batch(self, texts: List[str]) -> Dict[str, List[int]]:
        """Batch prediction"""
        results = {}
        
        for model_info in self.models.values():
            try:
                predictions = model_info['model'].predict(texts)
                results[model_info['display_name']] = predictions
            except Exception as e:
                print(f"Model {model_info['display_name']} prediction failed: {e}")
                results[model_info['display_name']] = [0] * len(texts)

        return results

    def ensemble_predict(self, text: str) -> Tuple[int, float]:
        """Ensemble prediction"""
        if len(self.models) < 2:
            raise ValueError("Ensemble prediction requires at least 2 models")
        
        results = self.predict_single(text)

        # Weighted average (using simple average here, weights can be adjusted based on model performance)
        total_weight = 0
        weighted_prob = 0
        
        for model_name, (pred, conf) in results.items():
            if conf > 0:  # Only consider valid predictions
                prob = conf if pred == 1 else 1 - conf
                weighted_prob += prob
                total_weight += 1
        
        if total_weight == 0:
            return 0, 0.5
        
        final_prob = weighted_prob / total_weight
        final_pred = int(final_prob > 0.5)
        final_conf = final_prob if final_pred == 1 else 1 - final_prob
        
        return final_pred, final_conf
    
    def _select_and_load_model(self):
        """Let user select and load model"""
        print("Qwen3 Weibo Sentiment Analysis Prediction System")
        print("="*40)
        print("Please select the model to use:")
        print("\nMethod Selection:")
        print("  1. Embedding + Classification Head (fast inference, less VRAM)")
        print("  2. LoRA Fine-tuning (better results, more VRAM)")
        
        method_choice = None
        while method_choice not in ['1', '2']:
            method_choice = input("\nPlease select method (1/2): ").strip()
            if method_choice not in ['1', '2']:
                print("Invalid choice, please enter 1 or 2")

        method_type = "embedding" if method_choice == '1' else "lora"
        method_name = "Embedding + Classification Head" if method_choice == '1' else "LoRA Fine-tuning"

        print(f"\nSelected: {method_name}")
        print("\nModel Size Selection:")
        print("  1. 0.6B - Lightweight, fast inference")
        print("  2. 4B  - Medium scale, balanced performance")
        print("  3. 8B  - Large scale, best performance")
        
        size_choice = None
        while size_choice not in ['1', '2', '3']:
            size_choice = input("\nPlease select model size (1/2/3): ").strip()
            if size_choice not in ['1', '2', '3']:
                print("Invalid choice, please enter 1, 2 or 3")

        size_map = {'1': '0.6B', '2': '4B', '3': '8B'}
        model_size = size_map[size_choice]

        print(f"Selected: Qwen3-{method_name}-{model_size}")
        print("Loading model...")
        
        try:
            self.load_model(method_type, model_size)
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Please check if model file exists, or train the model first")
    
    def interactive_predict(self):
        """Interactive prediction mode"""
        if len(self.models) == 0:
            # Let user select and load model
            self._select_and_load_model()
            if len(self.models) == 0:
                print("No models loaded, exiting prediction")
                return
        
        print("\n" + "="*60)
        print("Qwen3 Weibo Sentiment Analysis Prediction System")
        print("="*60)
        print("Loaded models:")
        for model_info in self.models.values():
            print(f"   - {model_info['display_name']}")
        print("\nCommand hints:")
        print("   Enter 'q' to exit program")
        print("   Enter 'switch' to switch model")
        print("   Enter 'models' to view loaded models")
        print("   Enter 'compare' to compare all model performance")
        print("-"*60)
        
        while True:
            try:
                text = input("\nPlease enter Weibo content to analyze: ").strip()

                if text.lower() == 'q':
                    print("Thank you for using, goodbye!")
                    break

                if text.lower() == 'models':
                    print("Loaded models:")
                    for model_info in self.models.values():
                        print(f"   - {model_info['display_name']}")
                    continue

                if text.lower() == 'switch':
                    print("Switching model...")
                    self.models.clear()  # Clear current models
                    self._select_and_load_model()
                    if len(self.models) > 0:
                        print("Model switch successful!")
                        for model_info in self.models.values():
                            print(f"   Current model: {model_info['display_name']}")
                    continue

                if text.lower() == 'compare':
                    test_text = input("Please enter text to compare: ")
                    self._compare_models(test_text)
                    continue

                if not text:
                    print("Please enter valid content")
                    continue
                
                # Prediction
                results = self.predict_single(text)

                print(f"\nOriginal text: {text}")
                print("Prediction results:")

                # Display sorted by model type and size
                sorted_results = sorted(results.items())
                for model_name, (pred, conf) in sorted_results:
                    sentiment = "Positive" if pred == 1 else "Negative"
                    print(f"   {model_name:20}: {sentiment} (Confidence: {conf:.4f})")

                # Only display single model prediction results (no ensemble)

            except KeyboardInterrupt:
                print("\n\nProgram interrupted, goodbye!")
                break
            except Exception as e:
                print(f"Error during prediction: {e}")
    
    def _compare_models(self, text: str):
        """Compare performance of different models"""
        print(f"\nModel Performance Comparison - Text: {text}")
        print("-" * 60)

        results = self.predict_single(text)

        embedding_models = []
        lora_models = []

        for model_name, (pred, conf) in results.items():
            sentiment = "Positive" if pred == 1 else "Negative"
            if "Embedding" in model_name:
                embedding_models.append((model_name, sentiment, conf))
            elif "Lora" in model_name:
                lora_models.append((model_name, sentiment, conf))

        if embedding_models:
            print("Embedding + Classification Head method:")
            for name, sentiment, conf in embedding_models:
                print(f"   {name}: {sentiment} ({conf:.4f})")

        if lora_models:
            print("LoRA Fine-tuning method:")
            for name, sentiment, conf in lora_models:
                print(f"   {name}: {sentiment} ({conf:.4f})")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Qwen3 Weibo Sentiment Analysis Unified Prediction Interface')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Model file directory')
    parser.add_argument('--model_type', type=str, choices=['embedding', 'lora'],
                        help='Specify model type')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'],
                        help='Specify model size')
    parser.add_argument('--text', type=str,
                        help='Directly predict specified text')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='Interactive prediction mode (default)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble prediction')
    parser.add_argument('--load_all', action='store_true',
                        help='Load all available models')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = Qwen3UniversalPredictor()
    
    # Load models
    if args.load_all:
        # Load all models
        predictor.load_all_models(args.model_dir)
    elif args.model_type and args.model_size:
        # Load specified model
        predictor.load_model(args.model_type, args.model_size)
    # If no model specified, interactive mode will let user choose

    # If text is specified, predict directly
    if args.text:
        if args.ensemble and len(predictor.models) > 1:
            pred, conf = predictor.ensemble_predict(args.text)
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {args.text}")
            print(f"Ensemble prediction: {sentiment} (Confidence: {conf:.4f})")
        else:
            results = predictor.predict_single(args.text)
            print(f"Text: {args.text}")
            for model_name, (pred, conf) in results.items():
                sentiment = "Positive" if pred == 1 else "Negative"
                print(f"{model_name}: {sentiment} (Confidence: {conf:.4f})")
    else:
        # Enter interactive mode
        predictor.interactive_predict()


if __name__ == "__main__":
    main()