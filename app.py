import os
import json
import math
import csv
import io
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models import db, User, StudentRecord
from src.adaptive.bkt_engine import BKTEngine, QUESTION_BANK, get_questions_for_concept
from src.models.explainability import generate_student_explanation

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "web", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.config['SECRET_KEY'] = 'mentora-ai-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'mentora.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

bkt_engine = BKTEngine(p_init=0.2, p_transit=0.15, p_slip=0.1, p_guess=0.2)


def init_db():
    """Initializes database and seeds default users and initial student cohort if empty."""
    with app.app_context():
        db.create_all()
        
        # Seed default users if none exist
        if not User.query.first():
            edu = User(username='educator1', password_hash=generate_password_hash('password123'), role='educator')
            stu = User(username='student1', password_hash=generate_password_hash('password123'), role='student', student_id_ref='OULAD-28491')
            db.session.add_all([edu, stu])
            db.session.commit()

        # Seed default student cohort if empty
        if not StudentRecord.query.first():
            default_cohort = [
                {"student_id": "OULAD-28491", "name": "Aarav Sharma", "module": "AAA (2013J)", "vle_clicks": 210, "quiz_avg": 38.5, "late_submissions": 3, "risk_prob": 0.89, "risk_level": "High Risk", "shap_driver": "VLE Clicks (-0.34), Quiz Avg (-0.29)", "recommended_concept": "Database Normalization (3NF)", "bkt_mastery": 0.22},
                {"student_id": "OULAD-31042", "name": "Priya Nair", "module": "BBB (2014B)", "vle_clicks": 340, "quiz_avg": 44.0, "late_submissions": 2, "risk_prob": 0.78, "risk_level": "High Risk", "shap_driver": "Quiz Avg (-0.28), VLE Clicks (-0.22)", "recommended_concept": "Relational Algebra & Keys", "bkt_mastery": 0.28},
                {"student_id": "OULAD-19403", "name": "Rohan Verma", "module": "DDD (2013J)", "vle_clicks": 620, "quiz_avg": 61.0, "late_submissions": 1, "risk_prob": 0.52, "risk_level": "Moderate Risk", "shap_driver": "VLE Clicks (-0.11), Quiz Avg (-0.08)", "recommended_concept": "Transaction ACID Properties", "bkt_mastery": 0.48},
                {"student_id": "OULAD-55820", "name": "Sneha Joshi", "module": "FFF (2014J)", "vle_clicks": 1420, "quiz_avg": 82.5, "late_submissions": 0, "risk_prob": 0.14, "risk_level": "Low Risk", "shap_driver": "High VLE Activity (+0.41)", "recommended_concept": "Explainable AI & BKT Analytics", "bkt_mastery": 0.76}
            ]
            for s in default_cohort:
                record = StudentRecord(**s)
                db.session.add(record)
            db.session.commit()


def evaluate_student_profile(student_id, name, module, vle_clicks, quiz_avg, late_submissions, weak_concept=None):
    vle_clicks = float(vle_clicks)
    quiz_avg = float(quiz_avg)
    late_submissions = int(late_submissions)

    baseline_clicks = 850.0
    baseline_quiz = 68.0

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

    shap_clicks = round(-click_diff * 0.25, 2)
    shap_quiz = round(-quiz_diff * 0.30, 2)
    shap_late = round(-late_submissions * 0.08, 2)

    shap_driver = (
        f"VLE Clicks ({'+' if shap_clicks >= 0 else ''}{shap_clicks}), "
        f"Quiz Avg ({'+' if shap_quiz >= 0 else ''}{shap_quiz}), "
        f"On-Time ({'+' if shap_late >= 0 else ''}{shap_late})"
    )

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
        "recommended_concept": weak_concept,
        "bkt_mastery": initial_mastery
    }


# ==================== AUTH ROUTES ====================

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')

        user = User.query.filter_by(username=username, role=role).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == 'educator':
                return redirect(url_for('educator_dashboard'))
            else:
                return redirect(url_for('student_workspace'))
        else:
            flash('Invalid username, password, or role selection.', 'danger')
            
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login_page'))


# ==================== APP VIEWS ====================

@app.route('/')
@login_required
def educator_dashboard():
    if current_user.role != 'educator':
        return redirect(url_for('student_workspace'))
        
    students = StudentRecord.query.all()
    # Convert SQLAlchemy objects to dicts for templates
    student_list = [s.__dict__ for s in students]
    return render_template('educator_dashboard.html', students=student_list, concepts=list(QUESTION_BANK.keys()))


@app.route('/student')
@login_required
def student_workspace():
    students = StudentRecord.query.all()
    student_list = [s.__dict__ for s in students]
    
    if current_user.role == 'student':
        selected = next((s for s in student_list if s["student_id"] == current_user.student_id_ref), student_list[0])
    else:
        student_id = request.args.get('student_id')
        selected = next((s for s in student_list if s["student_id"] == student_id), student_list[0])
        
    return render_template('student_workspace.html', student=selected, students=student_list)


@app.route('/quiz')
@login_required
def quiz_page():
    students = StudentRecord.query.all()
    student_list = [s.__dict__ for s in students]
    
    if current_user.role == 'student':
        selected = next((s for s in student_list if s["student_id"] == current_user.student_id_ref), student_list[0])
    else:
        student_id = request.args.get('student_id')
        selected = next((s for s in student_list if s["student_id"] == student_id), student_list[0])
        
    return render_template('quiz_tutor.html', student=selected, students=student_list, concepts=list(QUESTION_BANK.keys()))


# ==================== API ENDPOINTS ====================

@app.route('/api/v1/students/evaluate', methods=['POST'])
@login_required
def add_and_evaluate_student():
    try:
        data = request.get_json()
        student_id = data.get('student_id', '').strip() or f"STU-999"
        name = data.get('name', '').strip() or student_id
        module = data.get('module', 'DBMS (MCA-301)')
        vle_clicks = float(data.get('vle_clicks', 400))
        quiz_avg = float(data.get('quiz_avg', 50))
        late_submissions = int(data.get('late_submissions', 0))
        weak_concept = data.get('recommended_concept', 'AUTO')

        evaluated = evaluate_student_profile(
            student_id, name, module, vle_clicks, quiz_avg, late_submissions, weak_concept
        )

        existing = StudentRecord.query.filter_by(student_id=student_id).first()
        if existing:
            for k, v in evaluated.items():
                setattr(existing, k, v)
        else:
            record = StudentRecord(**evaluated)
            db.session.add(record)
        db.session.commit()

        return jsonify({"status": "success", "student": evaluated}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/v1/students/upload-csv', methods=['POST'])
@login_required
def upload_student_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded."}), 400
        file = request.files['file']
        
        stream = io.TextIOWrapper(file.stream, encoding='utf-8')
        reader = csv.DictReader(stream)

        count = 0
        for row in reader:
            s_id = row.get('student_id') or row.get('Student ID') or f"CSV-{count+100}"
            name = row.get('name') or row.get('Name') or s_id
            mod = row.get('module') or row.get('Module') or 'Batch CSV'
            clicks = float(row.get('vle_clicks') or 400)
            q_avg = float(row.get('quiz_avg') or 50)
            late = int(row.get('late_submissions') or 0)
            concept = row.get('recommended_concept') or 'AUTO'

            evaluated = evaluate_student_profile(s_id, name, mod, clicks, q_avg, late, concept)

            existing = StudentRecord.query.filter_by(student_id=s_id).first()
            if existing:
                for k, v in evaluated.items():
                    setattr(existing, k, v)
            else:
                db.session.add(StudentRecord(**evaluated))
            count += 1
            
        db.session.commit()
        return jsonify({"status": "success", "message": f"Successfully imported {count} students into database."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/v1/quiz/submit', methods=['POST'])
@login_required
def submit_quiz_answer():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        current_mastery = float(data.get('current_mastery', 0.2))
        correct = bool(data.get('correct', False))

        updated_mastery = bkt_engine.update_mastery(current_mastery, correct)
        status = "Mastered" if updated_mastery >= 0.8 else "In Progress"

        record = StudentRecord.query.filter_by(student_id=student_id).first()
        if record:
            record.bkt_mastery = updated_mastery
            db.session.commit()

        return jsonify({
            "status": "success",
            "previous_mastery": current_mastery,
            "correct": correct,
            "updated_mastery": updated_mastery,
            "concept_status": status
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/v1/tutor/chat', methods=['POST'])
@login_required
def tutor_chat():
    try:
        data = request.get_json()
        query = data.get('query', '').lower().strip()
        if '3nf' in query or 'bcnf' in query:
            answer = "Third Normal Form (3NF) eliminates transitive functional dependencies."
            citation = "Course PDF: Database Normalization (Unit 2, pp. 14–19)"
        else:
            answer = f"Based on your query regarding '{query}', the course emphasizes continuous engagement."
            citation = "Course PDF: Mentora AI Curriculum (p. 3)"
        return jsonify({"status": "success", "query": query, "answer": answer, "citation": citation}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    init_db()
    print("[INFO] Starting Mentora AI with Persistent SQLite DB & Auth on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)