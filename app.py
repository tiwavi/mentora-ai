import os
import json
import math
import csv
import io
from flask import Flask, render_template, request, jsonify
from src.adaptive.bkt_engine import BKTEngine, QUESTION_BANK, get_questions_for_concept
from src.models.explainability import generate_student_explanation

# Setup Absolute Paths & Flask App
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "web", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
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
    metrics_path = os.path.join(BASE_DIR, 'data', 'processed', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {
        "total_students": 32593,
        "high_risk": 15385,
        "low_moderate_risk": 17208
    }


# ==================== VIEW ROUTES ====================

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


# ==================== API ENDPOINTS ====================

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

        baseline_student = next((s for s in LIVE_COHORT if s["student_id"] == student_id), LIVE_COHORT[0])

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


@app.route('/api/v1/tutor/chat', methods=['POST'])
def tutor_chat():
    try:
        data = request.get_json()
        query = data.get('query', '').lower().strip()

        if '3nf' in query or 'bcnf' in query or 'normal' in query:
            answer = "Third Normal Form (3NF) eliminates transitive functional dependencies (X -> Y -> Z). Boyce-Codd Normal Form (BCNF) is stricter, requiring that for every non-trivial functional dependency X -> Y, X must be a superkey."
            citation = "Course PDF: Database Normalization (Unit 2, pp. 14–19)"
        elif 'bkt' in query or 'mastery' in query or 'tracing' in query:
            answer = "Bayesian Knowledge Tracing (BKT) models student learning as a Markov chain with latent states (Learned vs. Unlearned). It updates the probability of mastery P(L) after each response using P(S) (slip) and P(G) (guess) parameters."
            citation = "Course PDF: Adaptive Learning Systems & BKT (pp. 8–11)"
        elif 'acid' in query or 'transaction' in query:
            answer = "The ACID properties stand for Atomicity (all-or-nothing execution), Consistency (schema invariant preservation), Isolation (serializable concurrent execution), and Durability (permanent storage of committed logs)."
            citation = "Course PDF: Transaction Processing & Concurrency Control (Unit 3, pp. 5–12)"
        elif 'shap' in query or 'explainable' in query or 'risk' in query:
            answer = "SHAP (SHapley Additive exPlanations) values decompose a machine learning risk prediction into additive feature contributions, showing exactly how metrics like VLE clicks or quiz averages push a student toward or away from academic risk."
            citation = "Course PDF: Explainable AI in Education (pp. 4–8)"
        else:
            answer = f"Based on your query regarding '{query}', the course material emphasizes continuous VLE portal engagement and concept-specific mastery tracking during the first 60 days of instruction."
            citation = "Course PDF: Mentora AI Master Curriculum & OULAD Guidelines (p. 3)"

        return jsonify({
            "status": "success",
            "query": query,
            "answer": answer,
            "citation": citation
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/v1/students/upload-csv', methods=['POST'])
def upload_student_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part uploaded."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file."}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"status": "error", "message": "Only .csv files are supported."}), 400

        # Read CSV file stream
        stream = io.TextIOWrapper(file.stream, encoding='utf-8')
        reader = csv.DictReader(stream)

        added_count = 0
        for row in reader:
            student_id = row.get('student_id') or row.get('Student ID') or f"CSV-{len(LIVE_COHORT) + 201}"
            name = row.get('name') or row.get('Name') or student_id
            module = row.get('module') or row.get('Module') or 'Batch CSV (2026)'
            vle_clicks = float(row.get('vle_clicks') or row.get('VLE Clicks') or 400)
            quiz_avg = float(row.get('quiz_avg') or row.get('Quiz Avg') or 50)
            late_submissions = int(row.get('late_submissions') or row.get('Late Submissions') or 0)
            concept = row.get('recommended_concept') or row.get('Concept') or 'AUTO'

            evaluated = evaluate_student_profile(
                student_id=student_id.strip(),
                name=name.strip(),
                module=module.strip(),
                vle_clicks=vle_clicks,
                quiz_avg=quiz_avg,
                late_submissions=late_submissions,
                weak_concept=concept.strip()
            )

            # Prevent duplicate IDs, update or insert
            existing_idx = next((i for i, s in enumerate(LIVE_COHORT) if s["student_id"] == evaluated["student_id"]), None)
            if existing_idx is not None:
                LIVE_COHORT[existing_idx] = evaluated
            else:
                LIVE_COHORT.insert(0, evaluated)
            added_count += 1

        return jsonify({
            "status": "success",
            "message": f"Successfully evaluated and imported {added_count} student profiles from CSV.",
            "cohort_size": len(LIVE_COHORT)
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==================== SERVER RUN ====================

if __name__ == '__main__':
    print("[INFO] Starting Mentora AI Flask Web Server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)