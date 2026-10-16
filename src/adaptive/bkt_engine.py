"""
Module: bkt_engine.py
Author: Aryan (Team Member - Adaptive Learning & AI Tutor)
Description: Implements Bayesian Knowledge Tracing (BKT) to update student 
             concept mastery probabilities based on quiz response correctness.
"""

class BKTEngine:
    def __init__(self, p_init=0.2, p_transit=0.1, p_slip=0.1, p_guess=0.2):
        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update_mastery(self, p_learned, correct):
        if correct:
            p_l_given_obs = (p_learned * (1.0 - self.p_slip)) / (
                (p_learned * (1.0 - self.p_slip)) + ((1.0 - p_learned) * self.p_guess)
            )
        else:
            p_l_given_obs = (p_learned * self.p_slip) / (
                (p_learned * self.p_slip) + ((1.0 - p_learned) * (1.0 - self.p_guess))
            )
            
        p_next_learned = p_l_given_obs + (1.0 - p_l_given_obs) * self.p_transit
        return round(p_next_learned, 4)