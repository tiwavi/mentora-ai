"""
Module: cutoff_aggregator.py
Author: Shreyansh Tiwari (Team Lead - ML & Backend)
Description: Ingests OULAD raw tables, removes temporal leakage columns, 
             imputes missing values, and aggregates VLE clicks and assessment 
             scores strictly up to an early-course cutoff day T (e.g., Day 60).
"""

import os
import pandas as pd
import numpy as np

def load_raw_oulad(raw_dir="data/raw"):
    "" "Loads the core 6 OULAD CSV tables."" "
    print("[INFO] Loading raw OULAD tables...")
    try:
        courses = pd.read_csv(os.path.join(raw_dir, "courses.csv"))
        studentInfo = pd.read_csv(os.path.join(raw_dir, "studentInfo.csv"))
        studentRegistration = pd.read_csv(os.path.join(raw_dir, "studentRegistration.csv"))
        studentAssessment = pd.read_csv(os.path.join(raw_dir, "studentAssessment.csv"))
        assessments = pd.read_csv(os.path.join(raw_dir, "assessments.csv"))
        studentVle = pd.read_csv(os.path.join(raw_dir, "studentVle.csv"))
        return courses, studentInfo, studentRegistration, studentAssessment, assessments, studentVle
    except FileNotFoundError as e:
        print(f"[ERROR] Missing raw CSV file in {raw_dir}: {e}")
        print("[TIP] Place your raw OULAD CSVs in data/raw/ before running this script.")
        return None

def build_cutoff_dataset(cutoff_day=60, raw_dir="data/raw", output_dir="data/processed"):
    "" "
    Builds a leak-free feature matrix filtered up to 'cutoff_day'.
    "" "
    tables = load_raw_oulad(raw_dir)
    if tables is None:
        print("[WARN] Raw OULAD tables not found. Using pre-cleaned dataset if available.")
        return
    
    courses, studentInfo, studentRegistration, studentAssessment, assessments, studentVle = tables
    print(f"[INFO] Building feature matrix with early-course cutoff T = {cutoff_day} days...")

    # 1. Clean studentInfo
    studentInfo['imd_band'] = studentInfo['imd_band'].fillna('Unknown')
    studentInfo['disability'] = studentInfo['disability'].map({'N': 0, 'Y': 1})

    # 2. Prevent Leakage: Drop unregistration date
    if 'date_unregistration' in studentRegistration.columns:
        studentRegistration = studentRegistration.drop(columns=['date_unregistration'])

    # 3. Aggregate VLE clicks strictly up to cutoff_day
    vle_filtered = studentVle[studentVle['date'] <= cutoff_day]
    vle_agg = vle_filtered.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        vle_total_clicks_T=('sum_click', 'sum'),
        vle_active_days_T=('date', 'nunique')
    ).reset_index()

    # 4. Aggregate Assessments submitted on or before cutoff_day
    ass_merged = studentAssessment.merge(assessments, on='id_assessment', how='inner')
    ass_filtered = ass_merged[ass_merged['date_submitted'] <= cutoff_day]
    
    ass_agg = ass_filtered.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        quiz_weighted_score_T=('score', 'mean'),
        quiz_submission_count_T=('id_assessment', 'count')
    ).reset_index()

    # 5. Merge base records
    base_df = studentInfo.merge(
        studentRegistration[['code_module', 'code_presentation', 'id_student']], 
        on=['code_module', 'code_presentation', 'id_student'], 
        how='inner'
    )

    # 6. Join VLE and Assessment aggregates
    final_df = base_df.merge(vle_agg, on=['code_module', 'code_presentation', 'id_student'], how='left')
    final_df = final_df.merge(ass_agg, on=['code_module', 'code_presentation', 'id_student'], how='left')

    final_df['vle_total_clicks_T'] = final_df['vle_total_clicks_T'].fillna(0)
    final_df['vle_active_days_T'] = final_df['vle_active_days_T'].fillna(0)
    final_df['quiz_weighted_score_T'] = final_df['quiz_weighted_score_T'].fillna(0.0)
    final_df['quiz_submission_count_T'] = final_df['quiz_submission_count_T'].fillna(0)

    result_mapping = {'Pass': 0, 'Distinction': 0, 'Fail': 1, 'Withdrawn': 1}
    final_df['risk_target'] = final_df['final_result'].map(result_mapping)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cutoff_dataset_day{cutoff_day}.csv")
    final_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Processed cutoff dataset saved to {out_path}")

if __name__ == "__main__":
    build_cutoff_dataset(cutoff_day=60)
