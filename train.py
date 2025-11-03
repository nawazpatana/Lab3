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
    #H:\\NSU\\Lab3\\data\\spam.csv
    try:
        # Load the data - using your exact path
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
                'test_size': 0.2,
                'random_state': 42,
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

def run_logistic_regression_experiment():
    """Run Logistic Regression experiment with MLflow tracking"""
    print("\n" + "="*50)
    print("EXPERIMENT 2: LOGISTIC REGRESSION")
    print("="*50)
    
    try:
        with mlflow.start_run(run_name="logistic_regression"):
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
                'model_type': 'logistic_regression',
                'max_features': 1000,
                'test_size': 0.2,
                'random_state': 42,
                'max_iter': 1000,
                'dataset_size': len(df)
            })
            
            # Initialize and train Logistic Regression
            lr_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
            
            # Train model
            training_start = time.time()
            print("🔄 Training Logistic Regression...")
            lr_model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # Make predictions
            y_pred = lr_model.predict(X_test)
            y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
            
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
            cm_plot_path = plot_confusion_matrix(y_test, y_pred, "Logistic Regression")
            if cm_plot_path:
                mlflow.log_artifact(cm_plot_path)
            
            # Log model
            mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
            
            # Save model locally
            model_path = 'models/logistic_regression_model.pkl'
            joblib.dump(lr_model, model_path)
            mlflow.log_artifact(model_path)
            print(f"✅ Model saved to: {model_path}")
            
            # Save metrics
            metrics = {
                'model': 'logistic_regression',
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'training_time': float(training_time),
                'total_time': float(total_time),
                'dataset_size': len(df),
                'test_size': len(y_test)
            }
            
            # Create metrics directory if it doesn't exist
            os.makedirs('metrics', exist_ok=True)
            with open('metrics/lr_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            mlflow.log_artifact('metrics/lr_metrics.json')
            
            print("✅ Logistic Regression experiment completed successfully!")
            return metrics, lr_model
            
    except Exception as e:
        print(f"❌ Error in Logistic Regression experiment: {e}")
        return None, None

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