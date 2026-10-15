"""
Module: train_classifiers.py
Author: Shreyansh Tiwari (Team Lead - ML & Backend)
Description: Trains and benchmarks Logistic Regression, Decision Tree, and 
             Random Forest classifiers on cleaned_ou_data.csv using stratified 
             train-test splits and exports performance metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def train_and_evaluate_models(data_path="cleaned_ou_data.csv", output_dir="data/processed"):
    print("[INFO] Loading cleaned OULAD dataset...")
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found in working directory.")
        return
    
    df = pd.read_csv(data_path)
    
    # Define features and target
    drop_cols = ['code_module', 'code_presentation', 'id_student', 'final_result', 'target_binary']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['target_binary']
    
    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"[INFO] Features -> Numerical: {len(numerical_cols)} | Categorical: {len(categorical_cols)}")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    evaluation_results = {}
    
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        roc_auc = float(roc_auc_score(y_test, y_prob))
        f1 = float(f1_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        evaluation_results[name] = {
            "roc_auc": round(roc_auc, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "confusion_matrix": cm
        }
        print(f"[RESULT {name}] ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")
        
    # Save metrics to JSON for API and report integration
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    print(f"[SUCCESS] Model performance metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate_models()