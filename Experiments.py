import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
import json
import os

def load_and_preprocess_data():
    """Load and preprocess the spam dataset"""
    print("Loading and preprocessing data...")
    
    # Load the data
    df = pd.read_csv('\DCS-NSU\DVCLab2\Data\spam.csv', encoding='latin-1')
    
    # Clean the data - keep only relevant columns and remove empty ones
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    
    # Remove any rows with missing values
    df = df.dropna()
    
    # Convert labels to binary (ham=0, spam=1)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset loaded: {len(df)} messages")
    print(f"Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
    
    return df

def run_random_forest_experiment():
    """Run Random Forest experiment for approximately 5 seconds"""
    print("\n" + "="*50)
    print("EXPERIMENT 1: RANDOM FOREST")
    print("="*50)
    
    start_time = time.time()
    
    # Load data
    df = load_and_preprocess_data()
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize and train Random Forest with time limit
    rf_model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train for approximately 4 seconds
    training_start = time.time()
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    training_time = time.time() - training_start
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save metrics
    metrics = {
        'model': 'random_forest',
        'accuracy': float(accuracy),
        'training_time': float(training_time),
        'total_time': float(total_time),
        'dataset_size': len(df),
        'test_size': len(y_test)
    }
    
    # Create metrics directory if it doesn't exist
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/rf_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def run_logistic_regression_experiment():
    """Run Logistic Regression experiment for approximately 5 seconds"""
    print("\n" + "="*50)
    print("EXPERIMENT 2: LOGISTIC REGRESSION")
    print("="*50)
    
    start_time = time.time()
    
    # Load data
    df = load_and_preprocess_data()
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize and train Logistic Regression with time limit
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        n_jobs=-1  # Use all available cores
    )
    
    # Train for approximately 4 seconds
    training_start = time.time()
    print("Training Logistic Regression...")
    lr_model.fit(X_train, y_train)
    training_time = time.time() - training_start
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save metrics
    metrics = {
        'model': 'logistic_regression',
        'accuracy': float(accuracy),
        'training_time': float(training_time),
        'total_time': float(total_time),
        'dataset_size': len(df),
        'test_size': len(y_test)
    }
    
    # Create metrics directory if it doesn't exist
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/lr_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    """Run both experiments and compare results"""
    print("SMS Spam Detection Experiments")
    print("Running each experiment for approximately 5 seconds...")
    
    # Run experiments
   # rf_results = run_random_forest_experiment()
    lr_results = run_logistic_regression_experiment()
    
    # Compare results
    print("\n" + "="*50)
    print("EXPERIMENT COMPARISON")
    print("="*50)
    print(f"Random Forest:")
   # print(f"  - Accuracy: {rf_results['accuracy']:.4f}")
   # print(f"  - Training Time: {rf_results['training_time']:.2f}s")
   # print(f"  - Total Time: {rf_results['total_time']:.2f}s")
    
    print(f"\nLogistic Regression:")
    print(f"  - Accuracy: {lr_results['accuracy']:.4f}")
    print(f"  - Training Time: {lr_results['training_time']:.2f}s")
    print(f"  - Total Time: {lr_results['total_time']:.2f}s")
    
    # print(f"\nComparison:")
    # if rf_results['accuracy'] > lr_results['accuracy']:
    #     accuracy_diff = rf_results['accuracy'] - lr_results['accuracy']
    #     print(f"  ✓ Random Forest is better by {accuracy_diff:.4f} accuracy")
    # elif lr_results['accuracy'] > rf_results['accuracy']:
    #     accuracy_diff = lr_results['accuracy'] - rf_results['accuracy']
    #     print(f"  ✓ Logistic Regression is better by {accuracy_diff:.4f} accuracy")
    # else:
    #     print("  ⚠ Both models have the same accuracy")
    
    # if rf_results['training_time'] < lr_results['training_time']:
    #     print(f"  ✓ Random Forest is faster to train")
    # else:
    #     print(f"  ✓ Logistic Regression is faster to train")

if __name__ == "__main__":
    main()