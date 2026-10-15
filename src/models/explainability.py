"""
Module: explainability.py
Author: Shreyansh Tiwari (Team Lead - ML & Backend)
Description: Computes local SHAP feature attributions for individual student 
             risk predictions, returning plain-English explanations.
"""

import os
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def generate_student_explanation(student_id, data_path="cleaned_ou_data.csv"):
    """
    Generates local SHAP explanations for a specific student ID.
    Returns top risk-increasing and risk-mitigating features.
    """
    if not os.path.exists(data_path):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(data_path)
    
    # Locate student record
    student_row = df[df['id_student'] == int(student_id)]
    if student_row.empty:
        student_row = df.head(1)
        
    # Define columns to drop for feature matrix X
    drop_cols = ['code_module', 'code_presentation', 'id_student', 'final_result', 'target_binary']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Select categorical and numerical columns explicitly
    cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Train quick demonstrator RF on first 1000 rows for speed
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    rf.fit(X_processed[:1000], df['target_binary'][:1000])
    
    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(rf)
    
    # Transform target student row (X already has drop_cols removed)
    student_x = student_row.drop(columns=[col for col in drop_cols if col in student_row.columns])
    student_x_processed = preprocessor.transform(student_x)
    
    shap_values = explainer.shap_values(student_x_processed)
    
    # Handle binary classification shap output format across shap versions
    if isinstance(shap_values, list):
        s_vals = shap_values[1][0] # Positive class (risk = 1)
    elif len(shap_values.shape) == 3:
        s_vals = shap_values[0, :, 1]
    else:
        s_vals = shap_values[0]
        
    feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    
    # Combine feature names with SHAP values
    attributions = sorted(zip(feature_names, s_vals), key=lambda x: abs(x[1]), reverse=True)
    
    top_factors = []
    for feat, val in attributions[:5]:
        direction = "increased" if val > 0 else "decreased"
        top_factors.append({
            "feature": feat,
            "shap_value": round(float(val), 4),
            "impact": f"This factor {direction} predicted academic risk."
        })
        
    return {
        "student_id": int(student_id),
        "predicted_risk_probability": float(rf.predict_proba(student_x_processed)[0][1]),
        "top_attributions": top_factors
    }

if __name__ == "__main__":
    df_test = pd.read_csv("cleaned_ou_data.csv")
    sample_id = int(df_test.iloc[0]['id_student'])
    result = generate_student_explanation(sample_id)
    print("Explanation Test Result:", result)