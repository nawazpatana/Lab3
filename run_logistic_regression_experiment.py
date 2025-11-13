import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time
import json
import os
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def setup_mlflow():
    """Setup MLflow tracking for this experiment"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("spam-detection-experiment")
    print("✅ MLflow setup complete for Logistic Regression")

def load_and_preprocess_data():
    """Load and preprocess the spam dataset"""
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'message']
        df = df.dropna()
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        print(f"✅ Dataset loaded: {len(df)} messages")
        print(f"📊 Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_tfidf_features(df, max_features=1000):
    """Create TF-IDF features"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X = vectorizer.fit_transform(df['message'])
        y = df['label']
        
        # Save vectorizer
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        print("✅ TF-IDF features created and vectorizer saved")
        
        return X, y, vectorizer
        
    except Exception as e:
        print(f"❌ Error creating TF-IDF features: {e}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create and save confusion matrix"""
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
        
        plot_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ Confusion matrix saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"❌ Error creating confusion matrix: {e}")
        return None

def run_logistic_regression_experiment():
    """Run Logistic Regression experiment with MLflow tracking"""
    print("\n" + "="*50)
    print("🎯 EXPERIMENT: LOGISTIC REGRESSION (Running in MLflow)")
    print("="*50)
    
    # Setup MLflow for this experiment
    setup_mlflow()
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name="logistic_regression_experiment"):
            start_time = time.time()
            
            # Load data
            df = load_and_preprocess_data()
            if df is None:
                return None, None
            
            # Create features
            X, y, vectorizer = create_tfidf_features(df)
            if X is None:
                return None, None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Training set: {X_train.shape[0]} samples")
            print(f"📊 Test set: {X_test.shape[0]} samples")
            
            # Log parameters to MLflow
            mlflow.log_params({
                'model_type': 'logistic_regression',
                'max_features': 1000,
                'test_size': 0.2,
                'random_state': 42,
                'max_iter': 1000,
                'dataset_size': len(df)
            })
            
            # Train model
            lr_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
            
            training_start = time.time()
            print("🔄 Training Logistic Regression...")
            lr_model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # Predictions
            y_pred = lr_model.predict(X_test)
            y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            total_time = time.time() - start_time
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'training_time': training_time,
                'total_time': total_time,
                'precision_0': class_report['0']['precision'],
                'recall_0': class_report['0']['recall'],
                'f1_0': class_report['0']['f1-score'],
                'precision_1': class_report['1']['precision'],
                'recall_1': class_report['1']['recall'],
                'f1_1': class_report['1']['f1-score']
            })
            
            # Print results
            print(f"⏱️  Training time: {training_time:.2f} seconds")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"📈 ROC AUC: {roc_auc:.4f}")
            
            # Create and log confusion matrix
            cm_plot_path = plot_confusion_matrix(y_test, y_pred, "Logistic Regression")
            if cm_plot_path:
                mlflow.log_artifact(cm_plot_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
            
            # Save model locally
            model_path = 'models/logistic_regression_model.pkl'
            joblib.dump(lr_model, model_path)
            mlflow.log_artifact(model_path)
            
            # Save metrics locally
            metrics = {
                'model': 'logistic_regression',
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'training_time': float(training_time),
                'total_time': float(total_time),
                'dataset_size': len(df),
                'test_size': len(y_test)
            }
            
            os.makedirs('metrics', exist_ok=True)
            with open('metrics/lr_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            mlflow.log_artifact('metrics/lr_metrics.json')
            
            print("✅ Logistic Regression experiment completed and logged to MLflow!")
            print(f"🔗 Run ID: {mlflow.active_run().info.run_id}")
            
            return metrics, lr_model
            
    except Exception as e:
        print(f"❌ Error in Logistic Regression experiment: {e}")
        return None, None

if __name__ == "__main__":
    run_logistic_regression_experiment()