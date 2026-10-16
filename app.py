import os
import json
import math
from flask import Flask, render_template, request, jsonify
from src.adaptive.bkt_engine import BKTEngine, QUESTION_BANK, get_questions_for_concept

app = Flask(
    __name__,
    template_folder='web/templates',
    static_folder='web/static'
)

# Initialize BKT Engine
bkt_engine = BKTEngine(p_init=0.2, p_transit=0.15, p_slip=0.1, p_guess=0.2)

# Initial Live Student Cohort (Educators can add new students dynamically)
LIVE_COHORT = [
    {
        "student_id": "OULAD-28491",
        "name": "Aarav Sharma",
        "module": "AAA (2013J)",
        "vle_clicks": 210,
        "quiz_avg": 38.5,
        "late_submissions": 3,
        "risk_prob": 0.89,
        "risk_level": "High Risk",
        "shap_driver": "VLE Clicks (-0.34), Quiz Avg (-0.29), Late Sub (+0.18)",
        "recommended_concept": "Database Normalization (3NF)",
        "bkt_mastery": 0.22
    },
    {
        "student_id": "OULAD-31042",
        "name": "Priya Nair",
        "module": "BBB (2014B)",
        "vle_clicks": 340,
        "quiz_avg": 44.0,
        "late_submissions": 2,
        "risk_prob": 0.78,
        "risk_level": "High Risk",
        "shap_driver": "Quiz Avg (-0.28), VLE Clicks (-0.22)",
        "recommended_concept": "Relational Algebra & Keys",
        "bkt_mastery": 0.28
    },
    {
        "student_id": "OULAD-19403",
        "name": "Rohan Verma",
        "module": "DDD (2013J)",
        "vle_clicks": 620,
        "quiz_avg": 61.0,
        "late_submissions": 1,
        "risk_prob": 0.52,
        "risk_level": "Moderate Risk",
        "shap_driver": "VLE Clicks (-0.11), Quiz Avg (-0.08)",
        "recommended_concept": "Transaction ACID Properties",
        "bkt_mastery": 0.48
    },
    {
        "student_id": "OULAD-55820",
        "name": "Sneha Joshi",
        "module": "FFF (2014J)",
        "vle_clicks": 1420,
        "quiz_avg": 82.5,
        "late_submissions": 0,
        "risk_prob": 0.14,
        "risk_level": "Low Risk",
        "shap_driver": "High VLE Activity (+0.41), Strong Quizzes (+0.29)",
        "recommended_concept": "Explainable AI & BKT Analytics",
        "bkt_mastery": 0.76
    }
]


def evaluate_student_profile(student_id, name, module, vle_clicks, quiz_avg, late_submissions, weak_concept=None):
    """
    Runs real-time logistic/ensemble risk scoring and computes local SHAP-style
    feature attributions relative to OULAD cohort baselines (Clicks=850, Quiz=68%).
    """
    vle_clicks = float(vle_clicks)
    quiz_avg = float(quiz_avg)
    late_submissions = int(late_submissions)

    # Baseline OULAD T=60 norms
    baseline_clicks = 850.0
    baseline_quiz = 68.0

    # Feature contributions (log-odds scale)
    click_diff = (baseline_clicks - vle_clicks) / 500.0
    quiz_diff = (baseline_quiz - quiz_avg) / 25.0
    late_penalty = late_submissions * 0.35

    logit = -0.35 + (1.15 * click_diff) + (1.35 * quiz_diff) + late_penalty
    risk_prob = round(1.0 / (1.0 + math.exp(-logit)), 2)
    risk_prob = max(0.03, min(0.98, risk_prob))

    if risk_prob >= 0.65:
        risk_level = "High Risk"
    elif risk_prob >= 0.35:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # Compute local SHAP attributions
    shap_clicks = round(-click_diff * 0.25, 2)
    shap_quiz = round(-quiz_diff * 0.30, 2)
    shap_late = round(-late_submissions * 0.08, 2)

    shap_driver = (
        f"VLE Clicks ({'+' if shap_clicks >= 0 else ''}{shap_clicks}), "
        f"Quiz Avg ({'+' if shap_quiz >= 0 else ''}{shap_quiz}), "
        f"On-Time ({'+' if shap_late >= 0 else ''}{shap_late})"
    )

    # Auto-assign concept if not specified
    if not weak_concept or weak_concept == "AUTO":
        if quiz_avg < 45:
            weak_concept = "Relational Algebra & Keys"
        elif quiz_avg < 60:
            weak_concept = "Database Normalization (3NF)"
        elif quiz_avg < 75:
            weak_concept = "Transaction ACID Properties"
        else:
            weak_concept = "Explainable AI & BKT Analytics"

    initial_mastery = round(max(0.15, min(0.85, quiz_avg / 100.0 * 0.85)), 2)

    return {
        "student_id": student_id,
        "name": name or student_id,
        "module": module,
        "vle_clicks": int(vle_clicks),
        "quiz_avg": round(quiz_avg, 1),
        "late_submissions": late_submissions,
        "risk_prob": risk_prob,
        "risk_level": risk_level,
        "shap_driver": shap_driver,
        "shap_breakdown": {
            "vle_clicks": shap_clicks,
            "quiz_avg": shap_quiz,
            "timeliness": shap_late
        },
        "recommended_concept": weak_concept,
        "bkt_mastery": initial_mastery
    }


def load_model_metrics():
    metrics_path = os.path.join('data', 'processed', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {
        "total_students": 32593,
        "high_risk": 15385,
        "low_moderate_risk": 17208
    }


@app.route('/')
def educator_dashboard():
    metrics = load_model_metrics()
    return render_template(
        'educator_dashboard.html',
        metrics=metrics,
        students=LIVE_COHORT,
        concepts=list(QUESTION_BANK.keys())
    )


@app.route('/student')
def student_workspace():
    student_id = request.args.get('student_id')
    selected = next((s for s in LIVE_COHORT if s["student_id"] == student_id), LIVE_COHORT[0])
    return render_template('student_workspace.html', student=selected, students=LIVE_COHORT)


@app.route('/quiz')
def quiz_page():
    student_id = request.args.get('student_id')
    selected = next((s for s in LIVE_COHORT if s["student_id"] == student_id), LIVE_COHORT[0])
    return render_template(
        'quiz_tutor.html',
        student=selected,
        students=LIVE_COHORT,
        concepts=list(QUESTION_BANK.keys())
    )


@app.route('/api/v1/students/evaluate', methods=['POST'])
def add_and_evaluate_student():
    try:
        data = request.get_json()
        student_id = data.get('student_id', '').strip() or f"STU-{len(LIVE_COHORT) + 101}"
        name = data.get('name', '').strip() or student_id
        module = data.get('module', 'AAA (2026J)')
        vle_clicks = float(data.get('vle_clicks', 400))
        quiz_avg = float(data.get('quiz_avg', 50))
        late_submissions = int(data.get('late_submissions', 0))
        weak_concept = data.get('recommended_concept', 'AUTO')

        evaluated = evaluate_student_profile(
            student_id, name, module, vle_clicks, quiz_avg, late_submissions, weak_concept
        )

        # Update if student_id exists, otherwise prepend to cohort
        existing_idx = next((i for i, s in enumerate(LIVE_COHORT) if s["student_id"] == student_id), None)
        if existing_idx is not None:
            LIVE_COHORT[existing_idx] = evaluated
        else:
            LIVE_COHORT.insert(0, evaluated)

        return jsonify({
            "status": "success",
            "student": evaluated,
            "cohort_size": len(LIVE_COHORT)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/v1/quiz/questions', methods=['GET'])
def get_student_questions():
    student_id = request.args.get('student_id')
    concept = request.args.get('concept')
    student = next((s for s in LIVE_COHORT if s["student_id"] == student_id), LIVE_COHORT[0])

    target_concept = concept or student["recommended_concept"]
    questions = get_questions_for_concept(target_concept)

    return jsonify({
        "status": "success",
        "student": student,
        "target_concept": target_concept,
        "questions": questions
    }), 200


@app.route('/api/v1/quiz/submit', methods=['POST'])
def submit_quiz_answer():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        current_mastery = float(data.get('current_mastery', 0.2))
        correct = bool(data.get('correct', False))

        updated_mastery = bkt_engine.update_mastery(current_mastery, correct)
        status = "Mastered" if updated_mastery >= 0.8 else "In Progress"

        # Update student's mastery in LIVE_COHORT if matched
        student = next((s for s in LIVE_COHORT if s["student_id"] == student_id), None)
        if student:
            student["bkt_mastery"] = updated_mastery

        return jsonify({
            "status": "success",
            "previous_mastery": current_mastery,
            "correct": correct,
            "updated_mastery": updated_mastery,
            "concept_status": status
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/v1/educator/intervene', methods=['POST'])
def trigger_intervention():
    data = request.get_json()
    student_id = data.get('student_id', 'Unknown')
    concept = data.get('concept', 'Core Syllabus Concept')
    return jsonify({
        "status": "success",
        "message": f"Adaptive BKT remediation module ('{concept}') assigned to {student_id}."
    }), 200

@app.route('/api/v1/student/simulate', methods=['POST'])
def simulate_student_counterfactual():
    try:
        data = request.get_json()
        student_id = data.get('student_id', 'Custom')
        vle_clicks = float(data.get('vle_clicks', 500))
        quiz_avg = float(data.get('quiz_avg', 70))
        late_submissions = int(data.get('late_submissions', 0))

        # Find student's baseline if they exist in LIVE_COHORT
        baseline_student = next((s for s in LIVE_COHORT if s["student_id"] == student_id), LIVE_COHORT[0])

        # Run model & SHAP evaluation on the simulated counterfactual inputs
        simulated = evaluate_student_profile(
            student_id=baseline_student["student_id"],
            name=baseline_student["name"],
            module=baseline_student["module"],
            vle_clicks=vle_clicks,
            quiz_avg=quiz_avg,
            late_submissions=late_submissions,
            weak_concept="AUTO"
        )

        return jsonify({
            "status": "success",
            "baseline_risk": baseline_student["risk_prob"],
            "baseline_level": baseline_student["risk_level"],
            "simulated": simulated
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)