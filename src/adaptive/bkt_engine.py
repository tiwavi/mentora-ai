"""
Module: bkt_engine.py
Author: Aryan (Team Member - Adaptive Learning & AI Tutor)
Description: Implements Bayesian Knowledge Tracing (BKT) and provides a 
             concept-indexed adaptive question bank with grounded PDF excerpts.
"""

class BKTEngine:
    def __init__(self, p_init=0.2, p_transit=0.15, p_slip=0.1, p_guess=0.2):
        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update_mastery(self, p_learned, correct):
        """
        Updates the probability of mastery given a student's response (True/False).
        """
        p_learned = max(0.01, min(0.99, float(p_learned)))
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


# Concept-Indexed Adaptive Question Bank (12 Questions across 4 Syllabus Concepts)
QUESTION_BANK = {
    "Relational Algebra & Keys": [
        {
            "id": "RA-1",
            "concept": "Relational Algebra & Keys",
            "question": "Which relational algebra operator is used to select specific columns (attributes) from a relation?",
            "options": [
                {"text": "A) Selection (σ)", "correct": False},
                {"text": "B) Projection (π)", "correct": True},
                {"text": "C) Cartesian Product (×)", "correct": False},
                {"text": "D) Natural Join (⋈)", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 1, p. 9): 'The Projection operator (π) vertical-slices a relation, returning only the specified attribute columns and automatically eliminating duplicate tuples.'"
        },
        {
            "id": "RA-2",
            "concept": "Relational Algebra & Keys",
            "question": "What is the minimal superkey of a relation called?",
            "options": [
                {"text": "A) Foreign Key", "correct": False},
                {"text": "B) Surrogate Key", "correct": False},
                {"text": "C) Candidate Key", "correct": True},
                {"text": "D) Secondary Index", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 1, p. 14): 'A Candidate Key is a minimal superkey—meaning no proper subset of its attributes can uniquely identify a tuple in the relation.'"
        },
        {
            "id": "RA-3",
            "concept": "Relational Algebra & Keys",
            "question": "Referential integrity in a relational database requires that every non-null Foreign Key value must:",
            "options": [
                {"text": "A) Match an existing Primary Key value in the referenced relation", "correct": True},
                {"text": "B) Be unique across all tables in the schema", "correct": False},
                {"text": "C) Contain only numeric characters", "correct": False},
                {"text": "D) Be indexed using a B+ Tree", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 1, p. 22): 'The Referential Integrity Constraint states that a tuple in one relation referring to another relation must refer to an existing tuple in that referenced relation.'"
        }
    ],
    "Database Normalization (3NF)": [
        {
            "id": "NORM-1",
            "concept": "Database Normalization (3NF)",
            "question": "Which normal form specifically eliminates transitive functional dependencies of non-prime attributes on the primary key?",
            "options": [
                {"text": "A) First Normal Form (1NF)", "correct": False},
                {"text": "B) Second Normal Form (2NF)", "correct": False},
                {"text": "C) Third Normal Form (3NF)", "correct": True},
                {"text": "D) Unnormalized Form (UNF)", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 2, p. 14): 'A relation is in 3NF if it is in 2NF and no non-prime attribute is transitively dependent on the primary key (i.e., X -> Y and Y -> Z).'"
        },
        {
            "id": "NORM-2",
            "concept": "Database Normalization (3NF)",
            "question": "A relation is in Second Normal Form (2NF) if it is in 1NF and:",
            "options": [
                {"text": "A) It has no composite keys", "correct": False},
                {"text": "B) Every non-prime attribute is fully functionally dependent on the entire primary key", "correct": True},
                {"text": "C) It contains no foreign keys", "correct": False},
                {"text": "D) All attributes have atomic domains", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 2, p. 11): '2NF eliminates partial dependencies: no non-prime attribute may depend on only a proper subset of a composite candidate key.'"
        },
        {
            "id": "NORM-3",
            "concept": "Database Normalization (3NF)",
            "question": "In Boyce-Codd Normal Form (BCNF), for every non-trivial functional dependency X -> Y:",
            "options": [
                {"text": "A) Y must be a prime attribute", "correct": False},
                {"text": "B) X must be a superkey", "correct": True},
                {"text": "C) X and Y must be single attributes", "correct": False},
                {"text": "D) The table must have at least three columns", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 2, p. 19): 'BCNF is a stricter version of 3NF where every determinant X in a non-trivial dependency X -> Y must be a superkey.'"
        }
    ],
    "Transaction ACID Properties": [
        {
            "id": "ACID-1",
            "concept": "Transaction ACID Properties",
            "question": "Which ACID property guarantees that either all operations of a transaction are reflected in the database or none are?",
            "options": [
                {"text": "A) Atomicity", "correct": True},
                {"text": "B) Consistency", "correct": False},
                {"text": "C) Isolation", "correct": False},
                {"text": "D) Durability", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 3, p. 5): 'Atomicity (the all-or-nothing property) is managed by the transaction recovery subsystem using undo logs.'"
        },
        {
            "id": "ACID-2",
            "concept": "Transaction ACID Properties",
            "question": "Once a transaction commits, its changes persist even in the event of a system crash. Which property is this?",
            "options": [
                {"text": "A) Atomicity", "correct": False},
                {"text": "B) Isolation", "correct": False},
                {"text": "C) Durability", "correct": True},
                {"text": "D) Serializability", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 3, p. 7): 'Durability ensures that committed updates are written to non-volatile storage or write-ahead logs (WAL) before acknowledging commit.'"
        },
        {
            "id": "ACID-3",
            "concept": "Transaction ACID Properties",
            "question": "Which concurrency control protocol prevents deadlocks by ordering transactions using timestamps?",
            "options": [
                {"text": "A) Basic Two-Phase Locking (2PL)", "correct": False},
                {"text": "B) Wait-Die / Wound-Wait Timestamp Ordering", "correct": True},
                {"text": "C) Uncommitted Read", "correct": False},
                {"text": "D) Cartesian Locking", "correct": False}
            ],
            "excerpt": "Course PDF (Unit 3, p. 18): 'Wait-Die and Wound-Wait schemes use transaction timestamps to preemptively abort younger or older transactions, preventing circular wait cycles.'"
        }
    ],
    "Explainable AI & BKT Analytics": [
        {
            "id": "XAI-1",
            "concept": "Explainable AI & BKT Analytics",
            "question": "In Bayesian Knowledge Tracing (BKT), what does the parameter P(S) (Slip Probability) represent?",
            "options": [
                {"text": "A) Probability a student answers incorrectly despite having mastered the concept", "correct": True},
                {"text": "B) Probability a student guesses correctly without mastery", "correct": False},
                {"text": "C) Initial prior mastery before any quiz", "correct": False},
                {"text": "D) Dropout risk score", "correct": False}
            ],
            "excerpt": "Course PDF (Adaptive Systems, p. 8): 'P(S) captures human error or carelessness—when a learner who already knows the skill makes a mistake on an item.'"
        },
        {
            "id": "XAI-2",
            "concept": "Explainable AI & BKT Analytics",
            "question": "How do SHAP values help an educator interpret a student's academic risk prediction?",
            "options": [
                {"text": "A) They quantify how much each feature pushed the student's risk higher or lower than the cohort baseline", "correct": True},
                {"text": "B) They automatically expel high-risk students", "correct": False},
                {"text": "C) They replace the database schema", "correct": False},
                {"text": "D) They hide student engagement metrics", "correct": False}
            ],
            "excerpt": "Course PDF (XAI Guide, p. 4): 'SHAP additive explanations decompose a model prediction into individual feature contributions (e.g., VLE Clicks, Quiz Avg).'"
        },
        {
            "id": "XAI-3",
            "concept": "Explainable AI & BKT Analytics",
            "question": "Why is an early cutoff window (e.g., T = 60 days) used when training the OULAD risk classifier?",
            "options": [
                {"text": "A) To identify struggling students early enough in the semester for supportive intervention", "correct": True},
                {"text": "B) Because courses only last 60 days", "correct": False},
                {"text": "C) To reduce CSS file size", "correct": False},
                {"text": "D) To avoid using quiz scores", "correct": False}
            ],
            "excerpt": "Course PDF (Learning Analytics, p. 12): 'Early prediction cutoffs prevent data leakage from final exams and enable timely pedagogical support mid-semester.'"
        }
    ]
}

def get_questions_for_concept(concept=None):
    """Returns questions prioritized for the student's target concept, followed by other concepts."""
    if concept and concept in QUESTION_BANK:
        prioritized = list(QUESTION_BANK[concept])
        others = [q for c, qs in QUESTION_BANK.items() if c != concept for q in qs]
        return prioritized + others
    return [q for qs in QUESTION_BANK.values() for q in qs]