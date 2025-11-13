import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time
import json
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def setup_mlflow():
    """Setup MLflow tracking - use local only to avoid issues"""
    # Use local SQLite database to avoid authentication issues
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("spam-detection-experiment")
    print("✅ MLflow running locally with SQLite")
    print(f"📊 Experiment: spam-detection-experiment")

def load_and_preprocess_data():
    """Load and preprocess the spam dataset"""
    print("Loading and preprocessing data...")
    try:
        # Load the data
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        
        # Clean the data - keep only relevant columns and remove empty ones
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'message']
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Convert labels to binary (ham=0, spam=1)
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        print(f"✅ Dataset loaded: {len(df)} messages")
        print(f"📊 Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_tfidf_features(df, max_features=1000):
    """Create TF-IDF features and save vectorizer"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X = vectorizer.fit_transform(df['message'])
        y = df['label']
        
        # Save vectorizer for inference
        os.makedirs('models', exist_ok=True)
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        joblib.dump(vectorizer, vectorizer_path)
        print(f"✅ Vectorizer saved to: {vectorizer_path}")
        
        return X, y, vectorizer
        
    except Exception as e:
        print(f"❌ Error creating TF-IDF features: {e}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create and save confusion matrix plot"""
    try:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ Confusion matrix saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"❌ Error creating confusion matrix: {e}")
        return None

def register_best_model(rf_metrics, lr_metrics, rf_model, lr_model):
    """Register the best model based on accuracy"""
    
    if rf_metrics is None or lr_metrics is None:
        print("❌ Cannot register best model - one or both experiments failed")
        return None, None
    
    try:
        # Determine best model
        if rf_metrics['accuracy'] >= lr_metrics['accuracy']:
            best_model = rf_model
            best_model_name = "random_forest"
            best_accuracy = rf_metrics['accuracy']
            best_metrics = rf_metrics
        else:
            best_model = lr_model
            best_model_name = "logistic_regression"
            best_accuracy = lr_metrics['accuracy']
            best_metrics = lr_metrics
        
        # Register best model in MLflow
        with mlflow.start_run(run_name="best_model") as run:
            mlflow.log_params({
                'best_model': best_model_name,
                'best_accuracy': best_accuracy
            })
            
            mlflow.log_metrics({
                'accuracy': best_metrics['accuracy'],
                'roc_auc': best_metrics['roc_auc'],
                'training_time': best_metrics['training_time']
            })
            
            # Log best model
            mlflow.sklearn.log_model(best_model, "best_model")
            
            # Save best model locally
            best_model_path = 'models/best_model.pkl'
            joblib.dump(best_model, best_model_path)
            mlflow.log_artifact(best_model_path)
            
            # Save best model info
            best_model_info = {
                'model_name': best_model_name,
                'accuracy': best_accuracy,
                'registered_at': datetime.now().isoformat(),
                'run_id': run.info.run_id
            }
            
            with open('models/best_model_info.json', 'w') as f:
                json.dump(best_model_info, f, indent=2)
            
            mlflow.log_artifact('models/best_model_info.json')
            
            print(f"\n🏆 Best Model Registered: {best_model_name}")
            print(f"📊 Accuracy: {best_accuracy:.4f}")
            print(f"🔗 Run ID: {run.info.run_id}")
            
            return best_model, best_model_name
            
    except Exception as e:
        print(f"❌ Error registering best model: {e}")
        return None, None

def main():
    """Run both experiments and compare results"""
    print("🚀 SMS Spam Detection Experiments with MLflow")
    print("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    # Import and run experiments
    from run_random_forest_experiment import run_random_forest_experiment
    from run_logistic_regression_experiment import run_logistic_regression_experiment
    
    # Run experiments
    rf_results, rf_model = run_random_forest_experiment()
    lr_results, lr_model = run_logistic_regression_experiment()
    
    # Register best model only if both experiments succeeded
    if rf_results is not None and lr_results is not None:
        best_model, best_model_name = register_best_model(rf_results, lr_results, rf_model, lr_model)
        
        # Compare results
        print("\n" + "="*50)
        print("EXPERIMENT COMPARISON")
        print("="*50)
        print(f"Random Forest:")
        print(f"  - Accuracy: {rf_results['accuracy']:.4f}")
        print(f"  - ROC AUC: {rf_results['roc_auc']:.4f}")
        print(f"  - Training Time: {rf_results['training_time']:.2f}s")
        
        print(f"\nLogistic Regression:")
        print(f"  - Accuracy: {lr_results['accuracy']:.4f}")
        print(f"  - ROC AUC: {lr_results['roc_auc']:.4f}")
        print(f"  - Training Time: {lr_results['training_time']:.2f}s")
        
        print(f"\nComparison:")
        if rf_results['accuracy'] > lr_results['accuracy']:
            accuracy_diff = rf_results['accuracy'] - lr_results['accuracy']
            print(f"  ✓ Random Forest is better by {accuracy_diff:.4f} accuracy")
        elif lr_results['accuracy'] > rf_results['accuracy']:
            accuracy_diff = lr_results['accuracy'] - rf_results['accuracy']
            print(f"  ✓ Logistic Regression is better by {accuracy_diff:.4f} accuracy")
        else:
            print("  ⚠ Both models have the same accuracy")
        
        if rf_results['training_time'] < lr_results['training_time']:
            print(f"  ✓ Random Forest is faster to train")
        else:
            print(f"  ✓ Logistic Regression is faster to train")
        
        if best_model_name:
            print(f"\n🎯 Best Model Selected: {best_model_name}")
            print(f"📈 Best Accuracy: {max(rf_results['accuracy'], lr_results['accuracy']):.4f}")
    
    else:
        print("\n❌ One or both experiments failed. Check the errors above.")
        
    print("\n" + "="*60)
    print("🎉 Training completed! Next steps:")
    print("1. 🔍 Test predictions: python predict_spam.py")
    print("2. 🌐 View experiments: python start_mlflow_ui.py")
    print("="*60)

if __name__ == "__main__":
    main()