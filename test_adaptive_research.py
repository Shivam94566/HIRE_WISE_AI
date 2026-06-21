import unittest
import json
from app import app
from database.connection import db
from models.user import User
from models.question import Question
from models.interview import InterviewSession
from models.response import Response
from models.resume_upload import ResumeUpload
from models.skill_state import SessionSkillState, SessionSkillHistory
from services.adaptive_engine import AdaptiveEngine, SkillGraph
import services.adaptive_engine as adaptive_engine_module

class TestAdaptiveResearch(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  # Use isolated in-memory DB for unit tests
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Seed test user
        self.user = User(username="research_tester", email="tester@hirewise.ai")
        self.user.set_password("password123")
        db.session.add(self.user)
        db.session.commit()

        # Seed standard technical questions
        db.session.add(Question(text="Lists easy question", category="Technical", difficulty="Easy", skill="Python", subtopic="Lists"))
        db.session.add(Question(text="Lists hard question", category="Technical", difficulty="Hard", skill="Python", subtopic="Lists"))
        db.session.add(Question(text="Sorting medium question", category="Technical", difficulty="Medium", skill="DSA", subtopic="Sorting"))
        db.session.add(Question(text="Sorting hard question", category="Technical", difficulty="Hard", skill="DSA", subtopic="Sorting"))
        db.session.commit()
        
        self.engine = AdaptiveEngine()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_resume_aware_prior_initialization(self):
        """Verifies that starting priors are adjusted based on candidate resume content."""
        # 1. Test prior selection without resume
        session = InterviewSession(user_id=self.user.id, interview_type="Technical", status="started")
        db.session.add(session)
        db.session.commit()
        
        prior_score_no_resume, prior_unc_no_resume = self.engine.get_initial_prior_for_skill(session, "Python")
        self.assertEqual(prior_score_no_resume, 50.0)
        self.assertEqual(prior_unc_no_resume, 15.0)
        
        # 2. Test with resume upload content matching Python
        resume = ResumeUpload(
            user_id=self.user.id,
            filename="my_cv.pdf",
            file_path="uploads/resumes/my_cv.pdf",
            skills_extracted="Python, Django, Flask",
            projects="Built a web scraping pipeline in Python",
            experience="Senior Python Developer at Tech Corp",
            parsed_text="Highly proficient in python scripting and data analysis.",
            ats_score=85
        )
        db.session.add(resume)
        db.session.commit()
        
        # Prior score should scale based on skills & keywords evidence
        prior_score_with_resume, prior_unc_with_resume = self.engine.get_initial_prior_for_skill(session, "Python")
        self.assertGreater(prior_score_with_resume, 50.0)
        self.assertLess(prior_unc_with_resume, 15.0)
        print(f"[ TEST SUCCESS ] Resume-aware initialization: Prior Score = {prior_score_with_resume}, Uncertainty = {prior_unc_with_resume}")

    def test_skill_graph_propagation(self):
        """Verifies that a score update on a subtopic propagates to parents/prerequisites."""
        session = InterviewSession(user_id=self.user.id, interview_type="Technical", status="started")
        db.session.add(session)
        db.session.commit()
        
        # Initial score of subtopic 'Lists'
        lists_state = SessionSkillState(session_id=session.id, skill_name="Lists", score=50.0)
        db.session.add(lists_state)
        db.session.commit()
        
        # Propagate a positive update of +20.0 to 'Lists'
        self.engine.propagate_skill_graph_updates(session.id, "Lists", 20.0)
        
        # Verify parent node 'Python' is created and updated
        python_state = SessionSkillState.query.filter_by(session_id=session.id, skill_name="Python").first()
        self.assertIsNotNone(python_state)
        # Parent change is decay factor 0.5 * 20.0 = +10.0
        self.assertGreater(python_state.score, 50.0)
        print(f"[ TEST SUCCESS ] Skill propagation verified: Parent 'Python' updated to {python_state.score}")

    def test_bayesian_updates(self):
        """Verifies that Bayesian updates alter estimation scores and decrease uncertainty."""
        session = InterviewSession(user_id=self.user.id, interview_type="Technical", status="started")
        db.session.add(session)
        db.session.commit()
        
        # Perform initial response answer evaluation (simulated score: 95.0)
        state_after_update = self.engine.update_skill_state(session.id, "Python", 95.0)
        self.assertGreater(state_after_update.score, 50.0)
        
        # Check uncertainty parameter stored inside competency_map metadata
        c_map = self.engine._get_competency_map_metadata(session.id)
        unc_after_first = c_map.get("Python", {}).get("uncertainty")
        self.assertIsNotNone(unc_after_first)
        self.assertLess(unc_after_first, 15.0)
        
        # Perform second response update (simulated score: 90.0)
        self.engine.update_skill_state(session.id, "Python", 90.0)
        c_map2 = self.engine._get_competency_map_metadata(session.id)
        unc_after_second = c_map2.get("Python", {}).get("uncertainty")
        # Uncertainty should further decrease with additional observations
        self.assertLess(unc_after_second, unc_after_first)
        print(f"[ TEST SUCCESS ] Bayesian updates verified: Uncertainty decreased from {unc_after_first:.2f} to {unc_after_second:.2f}")

    def test_competency_convergence(self):
        """Verifies that the boundary convergence flag transitions to 'Converged' when uncertainty is low."""
        session = InterviewSession(user_id=self.user.id, interview_type="Technical", status="started")
        db.session.add(session)
        db.session.commit()
        
        # Run multiple simulated updates to push down standard error/uncertainty
        for i in range(6):
            self.engine.update_skill_state(session.id, "Python", 90.0)
            
        final_map = self.engine.detect_competency_boundaries(session.id)
        python_status = final_map.get("Python", {})
        self.assertEqual(python_status.get("confidence_level"), "Converged")
        print(f"[ TEST SUCCESS ] Competency convergence verified: Confidence Level is '{python_status.get('confidence_level')}'")

    def test_information_based_question_selection(self):
        """Verifies that item selection selects questions maximizing Fisher Information."""
        session = InterviewSession(user_id=self.user.id, interview_type="Technical", status="started")
        db.session.add(session)
        db.session.commit()
        
        # 1. Candidate is estimated at high capability (score = 85.0)
        state = SessionSkillState(session_id=session.id, skill_name="Lists", score=85.0)
        db.session.add(state)
        db.session.commit()
        
        # Information-theoretic selection should pick the 'Hard' difficulty question since it matches capability
        selected_q, _ = self.engine.get_next_question(session)
        self.assertEqual(selected_q.difficulty, "Hard")
        print(f"[ TEST SUCCESS ] Information-theoretic selection verified: Selected {selected_q.difficulty} question.")

    def test_feature_flags_toggle_rollback(self):
        """Verifies that turning off flags reverts logic back to the legacy system behavior."""
        session = InterviewSession(user_id=self.user.id, interview_type="Technical", status="started")
        db.session.add(session)
        db.session.commit()
        
        # Disable Bayesian Estimation flag
        adaptive_engine_module.ENABLE_BAYESIAN_ESTIMATION = False
        
        # Run update (should use legacy EMA: 0.4 * 90.0 + 0.6 * 50.0 = 66.0)
        state = self.engine.update_skill_state(session.id, "Python", 90.0)
        self.assertEqual(state.score, 66.0)
        
        # Re-enable flag
        adaptive_engine_module.ENABLE_BAYESIAN_ESTIMATION = True
        print("[ TEST SUCCESS ] Feature flags toggle/fallback verified.")

if __name__ == '__main__':
    unittest.main()
