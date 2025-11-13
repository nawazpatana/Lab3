import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time
import json
import os
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_random_forest_experiment():
    """Run Random Forest experiment with MLflow tracking"""
    print("\n" + "="*50)
    print("EXPERIMENT 1: RANDOM FOREST")
    print("="*50)
    
    try:
        with mlflow.start_run(run_name="random_forest"):
            start_time = time.time()
            
            # Load data
            df = load_and_preprocess_data()
            if df is None:
                return None, None
            
            # Create TF-IDF features
            X, y, vectorizer = create_tfidf_features(df)
            if X is None:
                return None, None
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Training set: {X_train.shape[0]} samples")
            print(f"📊 Test set: {X_test.shape[0]} samples")
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'random_forest',
                'n_estimators': 50,
                'max_features': 1000,
                'test_size': 0.4,
                'random_state': 40,
                'dataset_size': len(df)
            })
            
            # Initialize and train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            training_start = time.time()
            print("🔄 Training Random Forest...")
            rf_model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # Make predictions
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            total_time = time.time() - start_time
            
            # Log metrics
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
            print(f"⏱️  Total execution time: {total_time:.2f} seconds")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"📈 ROC AUC: {roc_auc:.4f}")
            
            # Create and log confusion matrix
            cm_plot_path = plot_confusion_matrix(y_test, y_pred, "Random Forest")
            if cm_plot_path:
                mlflow.log_artifact(cm_plot_path)
            
            # Log model
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            
            # Save model locally
            model_path = 'models/random_forest_model.pkl'
            joblib.dump(rf_model, model_path)
            mlflow.log_artifact(model_path)
            print(f"✅ Model saved to: {model_path}")
            
            # Save metrics
            metrics = {
                'model': 'random_forest',
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'training_time': float(training_time),
                'total_time': float(total_time),
                'dataset_size': len(df),
                'test_size': len(y_test)
            }
            
            # Create metrics directory if it doesn't exist
            os.makedirs('metrics', exist_ok=True)
            with open('metrics/rf_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            mlflow.log_artifact('metrics/rf_metrics.json')
            
            print("✅ Random Forest experiment completed successfully!")
            return metrics, rf_model
            
    except Exception as e:
        print(f"❌ Error in Random Forest experiment: {e}")
        return None, None

if __name__ == "__main__":
    # Setup MLflow for standalone execution
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("spam-detection-experiment")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Run the experiment
    run_random_forest_experiment()