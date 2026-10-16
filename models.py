"""
Module: models.py
Description: SQLAlchemy database models for Mentora AI persistent user authentication 
             and OULAD student risk records.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'educator' or 'student'
    student_id_ref = db.Column(db.String(50), nullable=True)  # Links student accounts to OULAD records


class StudentRecord(db.Model):
    __tablename__ = 'student_records'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(150), nullable=False)
    module = db.Column(db.String(100), nullable=False)
    vle_clicks = db.Column(db.Float, nullable=False)
    quiz_avg = db.Column(db.Float, nullable=False)
    late_submissions = db.Column(db.Integer, nullable=False)
    risk_prob = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    shap_driver = db.Column(db.String(255), nullable=True)
    recommended_concept = db.Column(db.String(150), nullable=True)
    bkt_mastery = db.Column(db.Float, nullable=False)