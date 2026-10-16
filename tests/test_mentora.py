"""
Module: test_mentora.py
Author: Aryan (Team Member - Adaptive Learning & Testing)
Description: Automated unit test suite verifying BKT mastery updates, ML risk 
             scoring calculations, and Flask API endpoint responses.
"""

import unittest
import json
from app import app, evaluate_student_profile
from src.adaptive.bkt_engine import BKTEngine, get_questions_for_concept

class MentoraTestCase(unittest.TestCase):
    
    def setUp(self):
        """Configure Flask test client before each test."""
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.bkt = BKTEngine(p_init=0.2, p_transit=0.15, p_slip=0.1, p_guess=0.2)

    def test_bkt_mastery_increase_on_correct(self):
        """Verify that answering correctly increases BKT mastery probability."""
        initial_p = 0.30
        updated_p = self.bkt.update_mastery(initial_p, correct=True)
        self.assertGreater(updated_p, initial_p, "Mastery probability should increase after a correct answer.")

    def test_bkt_mastery_decrease_on_incorrect(self):
        """Verify that answering incorrectly decreases BKT mastery probability."""
        initial_p = 0.50
        updated_p = self.bkt.update_mastery(initial_p, correct=False)
        self.assertLess(updated_p, initial_p, "Mastery probability should decrease after an incorrect answer.")

    def test_question_bank_retrieval(self):
        """Verify that the question bank returns prioritized questions for a given concept."""
        concept = "Database Normalization (3NF)"
        questions = get_questions_for_concept(concept)
        self.assertGreater(len(questions), 0, "Question bank should return questions.")
        self.assertEqual(questions[0]["concept"], concept, "First returned question should match the targeted concept.")

    def test_live_student_risk_evaluation(self):
        """Verify that live student evaluation correctly computes risk probabilities and SHAP attributes."""
        profile = evaluate_student_profile(
            student_id="TEST-001",
            name="Test Student",
            module="DBMS (MCA-301)",
            vle_clicks=150,
            quiz_avg=35.0,
            late_submissions=3,
            weak_concept="Database Normalization (3NF)"
        )
        self.assertEqual(profile["risk_level"], "High Risk", "Low clicks and low quiz scores should flag High Risk.")
        self.assertIn("vle_clicks", profile["shap_breakdown"], "SHAP breakdown should include vle_clicks feature attribution.")

    def test_flask_educator_dashboard_route(self):
        """Verify that the educator dashboard route returns HTTP 200."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200, "Educator dashboard route should load successfully.")

    def test_flask_tutor_chat_api(self):
        """Verify that the grounded PDF tutor chat endpoint responds correctly."""
        response = self.client.post('/api/v1/tutor/chat', 
                                  json={"query": "Explain 3NF vs BCNF"})
        self.assertEqual(response.status_code, 200, "Tutor chat API should return status 200.")
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("citation", data, "Tutor response should include a grounded PDF citation.")


if __name__ == '__main__':
    unittest.main()