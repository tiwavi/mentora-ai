"""
Module: bkt_engine.py
Author: Aryan (Team Member - Adaptive Learning & AI Tutor)
Description: Implements Bayesian Knowledge Tracing (BKT) to update student 
             concept mastery probabilities based on quiz response correctness.
"""

class BKTEngine:
    def __init__(self, p_init=0.2, p_transit=0.1, p_slip=0.1, p_guess=0.2):
        """
        Initializes the BKT parameters for a concept:
        - p_init: Initial probability that the student already knows the concept.
        - p_transit: Probability of transitioning from unlearned to learned after practice.
        - p_slip: Probability that a student knows the concept but answers incorrectly.
        - p_guess: Probability that a student does not know the concept but guesses correctly.
        """
        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update_mastery(self, p_learned, correct):
        """
        Updates the probability of mastery given a student's response (True/False).
        Uses Bayes' theorem to update belief, then applies the transition probability.
        """
        if correct:
            # P(L | Correct)
            p_l_given_obs = (p_learned * (1.0 - self.p_slip)) / (
                (p_learned * (1.0 - self.p_slip)) + ((1.0 - p_learned) * self.p_guess)
            )
        else:
            # P(L | Incorrect)
            p_l_given_obs = (p_learned * self.p_slip) / (
                (p_learned * self.p_slip) + ((1.0 - p_learned) * (1.0 - self.p_guess))
            )
            
        # Apply transition probability (learning curve progression)
        p_next_learned = p_l_given_obs + (1.0 - p_l_given_obs) * self.p_transit
        return round(p_next_learned, 4)

if __name__ == "__main__":
    # Test simulation of a student answering 3 questions on a concept
    bkt = BKTEngine()
    current_mastery = bkt.p_init
    print(f"Initial Concept Mastery: {current_mastery}")
    
    # Simulate student answers: Correct, Correct, Incorrect, Correct
    responses = [True, True, False, True]
    for i, ans in enumerate(responses, 1):
        current_mastery = bkt.update_mastery(current_mastery, ans)
        print(f"After Question {i} (Correct={ans}): Mastery Probability = {current_mastery}")