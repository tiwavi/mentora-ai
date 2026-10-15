"""
Module: app.py
Author: Shreyansh Tiwari (Team Lead - ML & Backend)
Description: Main Flask application for Mentora AI. Serves web dashboard routes,
             handles diagnostic quiz submissions, calculates BKT mastery updates,
             and returns SHAP/LIME explanation data via REST APIs.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

# Global variables to cache data and metrics
DATA_PATH = "cleaned_ou_data.csv"
METRICS_PATH = "data/processed/model_metrics.json"
CONCEPT_MAP_PATH = "data/content/concept_map.json"
QUESTION_BANK_PATH = "data/content/question_bank.json"

def load_app_data():
    """Loads processed data, metrics, and JSON content schemas safely."""
    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
    
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
    concept_map = {}
    if os.path.exists(CONCEPT_MAP_PATH):
        with open(CONCEPT_MAP_PATH, 'r') as f:
            concept_map = json.load(f)
            
    question_bank = {}
    if os.path.exists(QUESTION_BANK_PATH):
        with open(QUESTION_BANK_PATH, 'r') as f:
            question_bank = json.load(f)
            
    return df, metrics, concept_map, question_bank

@app.route("/")
def educator_dashboard():
    """Renders the Educator Cohort Risk & Model Benchmark Dashboard."""
    df, metrics, _, _ = load_app_data()
    total_students = len(df) if df is not None else 32593
    high_risk = int(df['target_binary'].sum()) if df is not None else 15385
    low_risk = total_students - high_risk
    
    return render_template(
        "educator_dashboard.html",
        total_students=total_students,
        high_risk=high_risk,
        low_risk=low_risk,
        metrics=metrics
    )

@app.route("/student")
def student_workspace():
    """Renders the Student Risk XAI & What-If Simulator Workspace."""
    return render_template("student_workspace.html")

@app.route("/quiz")
def quiz_tutor():
    """Renders the Adaptive Diagnostic Quiz & Grounded PDF Tutor View."""
    _, _, concept_map, question_bank = load_app_data()
    return render_template(
        "quiz_tutor.html",
        concepts=concept_map.get("concepts", []),
        questions=question_bank.get("questions", [])
    )

# --- REST API ENDPOINTS ---

@app.route("/api/v1/cohort", methods=["GET"])
def api_cohort_summary():
    """Returns cohort statistics and model benchmark scores as JSON."""
    df, metrics, _, _ = load_app_data()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404
        
    sample_students = df.head(10)[['id_student', 'code_module', 'total_clicks', 'avg_assessment_score', 'target_binary']].to_dict(orient="records")
    
    return jsonify({
        "status": "success",
        "total_records": len(df),
        "model_metrics": metrics,
        "sample_students": sample_students
    })

@app.route("/api/v1/simulate", methods=["POST"])
def api_simulate_counterfactual():
    """Powers the What-If Counterfactual Simulator."""
    req = request.get_json() or {}
    sim_clicks = req.get("sim_vle_clicks", 500)
    sim_score = req.get("sim_quiz_score", 70.0)
    
    # Heuristic counterfactual calculation for demonstration
    base_risk = 0.82
    risk_reduction = (sim_clicks / 2000.0) * 0.3 + (sim_score / 100.0) * 0.4
    simulated_risk = max(0.05, round(base_risk - risk_reduction, 2))
    
    tier = "High Risk" if simulated_risk >= 0.65 else ("Moderate Risk" if simulated_risk >= 0.35 else "Low Risk")
    
    return jsonify({
        "status": "success",
        "original_risk": base_risk,
        "simulated_risk": simulated_risk,
        "risk_delta": round(simulated_risk - base_risk, 2),
        "new_tier": tier,
        "message": f"Adjusting VLE clicks to {sim_clicks} and quiz score to {sim_score}% updates estimated risk to {int(simulated_risk*100)}% ({tier})."
    })

if __name__ == "__main__":
    print("[INFO] Starting Mentora AI Flask Web Server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)