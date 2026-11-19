import os
import uuid
import json
import random
import re
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from config import Config
from database.connection import db
from models.interview import InterviewSession
from models.question import Question
from models.response import Response
from services.groq_service import GroqService
from services.whisper_service import WhisperService
from services.cv_service import CVService
from services.email_service import EmailService

interview_bp = Blueprint('interview', __name__)

# Instantiate Services
groq_service = GroqService()
whisper_service = WhisperService()
cv_service = CVService()

@interview_bp.route('/interview/select', methods=['GET', 'POST'])
@login_required
def select():
    """Renders the screen to configure a new interview session."""
    if request.method == 'POST':
        interview_type = request.form.get('interview_type')
        company_name = request.form.get('company_name')
        role_applied = request.form.get('role_applied', 'Software Engineer').strip() or 'Software Engineer'
        experiment_mode = request.form.get('experiment_mode', 'fixed')
        
        # Create Interview Session in DB
        new_session = InterviewSession(
            user_id=current_user.id,
            interview_type=interview_type,
            company_name=company_name if interview_type == 'Company' else None,
            role_applied=role_applied,
            status='started',
            experiment_mode=experiment_mode
        )
        db.session.add(new_session)
        db.session.commit()
        
        # Increment total interviews on user stats
        current_user.total_interviews = (current_user.total_interviews or 0) + 1
        db.session.commit()
        
        # Select Questions
        question_ids = []
        if interview_type == 'Project':
            # Option A/B/C/D inputs
            project_option = request.form.get('project_input_option', 'github')
            project_text = ""
            
            project_inputs = {
                "option": project_option,
                "github_url": "",
                "deployed_url": "",
                "pdf_filename": "",
                "manual_title": "",
                "manual_description": "",
                "manual_technologies": "",
                "manual_features": "",
                "manual_contribution": "",
                "manual_challenges": ""
            }
            
            if project_option == 'github':
                github_url = request.form.get('project_github_url', '').strip()
                project_inputs["github_url"] = github_url
                project_text = f"GitHub Repository URL: {github_url}"
            elif project_option == 'deployed':
                deployed_url = request.form.get('project_deployed_url', '').strip()
                project_inputs["deployed_url"] = deployed_url
                project_text = f"Deployed Website URL: {deployed_url}"
            elif project_option == 'pdf':
                pdf_file = request.files.get('project_pdf')
                if pdf_file and pdf_file.filename != '':
                    filename = secure_filename(pdf_file.filename)
                    unique_filename = f"user_{current_user.id}_project_{filename}"
                    save_path = Config.UPLOAD_FOLDER / unique_filename
                    pdf_file.save(str(save_path))
                    
                    project_inputs["pdf_filename"] = filename
                    
                    from services.resume_service import ResumeService
                    resume_service = ResumeService()
                    pdf_text = resume_service.extract_text(save_path)
                    project_text = pdf_text
                else:
                    flash("Please upload a valid project synopsis PDF.", "warning")
                    return redirect(url_for('interview.select'))
            elif project_option == 'manual':
                title = request.form.get('project_title', '').strip()
                description = request.form.get('project_description', '').strip()
                technologies = request.form.get('project_technologies', '').strip()
                features = request.form.get('project_features', '').strip()
                contribution = request.form.get('project_contribution', '').strip()
                challenges = request.form.get('project_challenges', '').strip()
                
                project_inputs.update({
                    "manual_title": title,
                    "manual_description": description,
                    "manual_technologies": technologies,
                    "manual_features": features,
                    "manual_contribution": contribution,
                    "manual_challenges": challenges
                })
                
                project_text = f"Project Title: {title}\nDescription: {description}\nTechnologies: {technologies}\nFeatures: {features}\nPersonal Contribution: {contribution}\nChallenges Faced: {challenges}"
            
            if not project_text or len(project_text.strip()) < 5:
                flash("Insufficient project details provided.", "warning")
                return redirect(url_for('interview.select'))

            # Generate structured understanding JSON
            project_understanding = {
                "project_title": "Project",
                "domain": "Software Engineering",
                "technologies": [],
                "architecture": "Standard",
                "features": [],
                "probable_modules": [],
                "probable_challenges": [],
                "probable_scalability_topics": []
            }
            
            if groq_service.initialized:
                try:
                    prompt = f"""
You are a principal software architect and technical interviewer. Analyze the following candidate project information to extract a structured technical design breakdown:

Project Input:
{project_text}

Extract and synthesize the software architecture and engineering dimensions. Respond strictly with a valid JSON object matching this schema:
{{
  "project_title": "Precise title of the project",
  "domain": "Domain area (e.g. E-Commerce, Distributed Systems, FinTech, Machine Learning, Developer Tools)",
  "technologies": ["List of all detected programming languages, frameworks, libraries, databases, and deployment tools"],
  "architecture": "Clear description of the software architecture (e.g. Layered MVC, Event-Driven Microservices, Client-Server REST Monolith, Pipeline)",
  "features": ["Core functional capabilities implemented in the project"],
  "probable_modules": ["Logical system components (e.g. Authentication Service, Database Abstraction Layer, Async Worker, API Gateway)"],
  "probable_challenges": ["Real engineering bottlenecks or challenges (e.g. Concurrency race conditions, caching invalidation, PDF streaming latency, schema migrations)"],
  "probable_scalability_topics": ["Specific topics for scaling (e.g. Redis caching, read-replicas, rate limiting, connection pooling, indexing)"]
}}
Do not wrap in markdown code blocks. Respond strictly with valid JSON.
"""
                    llm_res = groq_service._create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        json_mode=True,
                        temperature=0.2,
                        max_tokens=900
                    )
                    llm_text = llm_res.choices[0].message.content.strip()
                    if llm_text.startswith("```"):
                        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_text)
                        if match:
                            llm_text = match.group(1).strip()
                    project_understanding = json.loads(llm_text)
                except Exception as ex:
                    print(f"[ WARNING ] Failed to generate project understanding via LLM: {ex}")
                    project_understanding["project_title"] = project_inputs.get("manual_title") or "My Project"
                    if project_inputs.get("manual_technologies"):
                        project_understanding["technologies"] = [t.strip() for t in project_inputs["manual_technologies"].split(",") if t.strip()]
            else:
                project_understanding["project_title"] = project_inputs.get("manual_title") or "My Project"
                if project_inputs.get("manual_technologies"):
                    project_understanding["technologies"] = [t.strip() for t in project_inputs["manual_technologies"].split(",") if t.strip()]

            # Generate questions pool: 21 questions (7 categories * 3 difficulties)
            questions_pool = []
            if groq_service.initialized:
                try:
                    prompt = f"""
You are an engineering hiring manager conducting a project-based technical interview round.
Read this verified architectural breakdown of the candidate's project:

{json.dumps(project_understanding, indent=2)}

Generate exactly 21 mock interview questions tailored directly to this project.
Cover each of the following 7 categories with exactly 3 questions (one Easy, one Medium, one Hard):
1. Project Understanding (High-level problem statement, target audience, core business value)
2. Architecture (Component relationships, data flow, API design, monolithic vs microservice choices)
3. Technology Stack (Why specific libraries/databases were chosen over industry alternatives)
4. Implementation (Code-level design patterns, state management, error handling, ORM/query design)
5. Challenges (Debugging incidents, performance bottlenecks, race conditions, schema trade-offs)
6. Scalability (Handling 10x traffic, caching strategies, database indexing, horizontal scaling)
7. Contribution Verification (Probing candidate's personal contribution vs template/boilerplate code)

Difficulty Guidelines:
- Easy: Checks basic familiarity with project features, purpose, and language/framework choices.
- Medium: Probes implementation mechanics, error recovery, state management, and architectural trade-offs.
- Hard: Stress-tests failure modes, concurrency, scalability bottlenecks, security, and edge-case engineering decisions.

Respond strictly with a JSON array of 21 objects. Do not add markdown code blocks or any other explanation. Each object must match:
{{
  "text": "The full, realistic interviewer question",
  "category": "Project",
  "skill": "The category name (Project Understanding, Architecture, Technology Stack, Implementation, Challenges, Scalability, or Contribution Verification)",
  "difficulty": "Easy, Medium, or Hard",
  "expected_keywords": ["keyword1", "keyword2", "keyword3"]
}}
"""
                    llm_res = groq_service._create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        json_mode=True,
                        temperature=0.3,
                        max_tokens=2500
                    )
                    llm_text = llm_res.choices[0].message.content.strip()
                    if llm_text.startswith("```"):
                        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_text)
                        if match:
                            llm_text = match.group(1).strip()
                    questions_pool = json.loads(llm_text)
                except Exception as ex:
                    print(f"[ WARNING ] Failed to generate project questions via LLM: {ex}")
            
            if not questions_pool:
                categories_list = [
                    "Project Understanding",
                    "Architecture",
                    "Technology Stack",
                    "Implementation",
                    "Challenges",
                    "Scalability",
                    "Contribution Verification"
                ]
                for cat in categories_list:
                    for diff in ["Easy", "Medium", "Hard"]:
                        questions_pool.append({
                            "text": f"Explain aspects of your project's {cat} at an {diff} level.",
                            "category": "Project",
                            "skill": cat,
                            "difficulty": diff,
                            "expected_keywords": ["project", cat.lower()]
                        })

            # Create temporary custom questions in the database
            project_all_ids = []
            for q_data in questions_pool:
                custom_q = Question(
                    text=q_data["text"],
                    category='Project',
                    skill=q_data["skill"],
                    difficulty=q_data["difficulty"],
                    expected_keywords=json.dumps(q_data.get("expected_keywords", ["project"]))
                )
                db.session.add(custom_q)
                db.session.commit()
                project_all_ids.append(custom_q.id)

            # Store project metadata in competency_map
            comp_map = {
                "project_inputs": project_inputs,
                "project_understanding": project_understanding,
                "project_all_question_ids": project_all_ids,
                "project_feedback": {}
            }
            new_session.competency_map = json.dumps(comp_map)
            db.session.commit()

            # Now assign question_ids for the session
            if experiment_mode != 'fixed':
                # Starting question: Project Understanding, Medium difficulty
                starting_q = db.session.query(Question).filter(
                    Question.id.in_(project_all_ids),
                    Question.skill == 'Project Understanding',
                    Question.difficulty == 'Medium'
                ).first()
                if not starting_q:
                    starting_q = db.session.query(Question).filter(Question.id.in_(project_all_ids)).first()
                
                if starting_q:
                    question_ids = [starting_q.id]
                    initial_log = [{
                        "question_index": 1,
                        "target_skill": starting_q.skill or "Project Understanding",
                        "target_difficulty": starting_q.difficulty or "Medium",
                        "selected_question_id": starting_q.id,
                        "selected_question_text": starting_q.text[:60] + "...",
                        "last_score": None,
                        "reason": "Initial project mock assessment starting question.",
                        "detected_skills": [
                            "Project Understanding", "Architecture", "Technology Stack", 
                            "Implementation", "Challenges", "Scalability", "Contribution Verification"
                        ],
                        "skill_priority": [
                            "Project Understanding", "Architecture", "Technology Stack", 
                            "Implementation", "Challenges", "Scalability", "Contribution Verification"
                        ],
                        "generated_interview_path": [starting_q.skill or "Project Understanding"]
                    }]
                    new_session.decision_log = json.dumps(initial_log)
            else:
                # Fixed mode: pick 5 questions from the pool, one from each of first 5 categories at Medium difficulty
                selected_qs = []
                categories_list = [
                    "Project Understanding",
                    "Architecture",
                    "Technology Stack",
                    "Implementation",
                    "Challenges"
                ]
                for cat in categories_list:
                    mq = db.session.query(Question).filter(
                        Question.id.in_(project_all_ids),
                        Question.skill == cat,
                        Question.difficulty == 'Medium'
                    ).first()
                    if not mq:
                        mq = db.session.query(Question).filter(
                            Question.id.in_(project_all_ids),
                            Question.skill == cat
                        ).first()
                    if mq:
                        selected_qs.append(mq)
                
                if len(selected_qs) < 5:
                    remaining_qs = db.session.query(Question).filter(
                        Question.id.in_(project_all_ids),
                        Question.id.notin_([q.id for q in selected_qs])
                    ).limit(5 - len(selected_qs)).all()
                    selected_qs.extend(remaining_qs)
                    
                question_ids = [q.id for q in selected_qs]

        elif experiment_mode != 'fixed':
            from services.adaptive_engine import AdaptiveEngine
            adaptive_engine = AdaptiveEngine()
            session_skills = adaptive_engine.get_skills_for_session(new_session)
            starting_skill = session_skills[0] if session_skills else 'Python'
            
            # Select initial question matching highest priority skill from resume
            starting_q = db.session.query(Question).filter_by(skill=starting_skill, difficulty='Medium').first()
            if not starting_q:
                starting_q = db.session.query(Question).filter(Question.skill == starting_skill).first()
            if not starting_q:
                starting_q = db.session.query(Question).first()
                
            if starting_q:
                question_ids = [starting_q.id]
                initial_log = [{
                    "question_index": 1,
                    "target_skill": starting_q.skill or "General",
                    "target_difficulty": starting_q.difficulty or "Medium",
                    "selected_question_id": starting_q.id,
                    "selected_question_text": starting_q.text[:60] + "...",
                    "last_score": None,
                    "reason": "Initial baseline technical competency assessment question.",
                    "detected_skills": session_skills,
                    "skill_priority": session_skills,
                    "generated_interview_path": [starting_q.skill or "General"]
                }]
                new_session.decision_log = json.dumps(initial_log)
            else:
                question_ids = []
        elif interview_type == 'HR':
            questions = db.session.query(Question).filter_by(category='HR').all()
            if not questions:
                questions = db.session.query(Question).all()
            sampled = random.sample(questions, min(5, len(questions)))
            question_ids = [q.id for q in sampled]
            
        elif interview_type == 'Technical':
            questions = db.session.query(Question).filter_by(category='Technical').all()
            if not questions:
                questions = db.session.query(Question).all()
            sampled = random.sample(questions, min(5, len(questions)))
            question_ids = [q.id for q in sampled]
            
        elif interview_type == 'Mixed':
            hr_qs = db.session.query(Question).filter_by(category='HR').all()
            tech_qs = db.session.query(Question).filter_by(category='Technical').all()
            behav_qs = db.session.query(Question).filter_by(category='Behavioral').all()
            
            sampled_hr = random.sample(hr_qs, min(2, len(hr_qs))) if hr_qs else []
            sampled_tech = random.sample(tech_qs, min(2, len(tech_qs))) if tech_qs else []
            sampled_behav = random.sample(behav_qs, min(1, len(behav_qs))) if behav_qs else []
            
            sampled = sampled_hr + sampled_tech + sampled_behav
            # Shuffle mixed questions
            random.shuffle(sampled)
            question_ids = [q.id for q in sampled]
            
        elif interview_type == 'Company':
            questions = db.session.query(Question).filter_by(category='Company', company=company_name).all()
            # If not enough company questions, pad with general Technical questions
            if len(questions) < 5:
                tech_qs = db.session.query(Question).filter_by(category='Technical').all()
                needed = 5 - len(questions)
                pad_qs = random.sample(tech_qs, min(needed, len(tech_qs))) if tech_qs else []
                questions.extend(pad_qs)
            sampled = questions[:5]
            question_ids = [q.id for q in sampled]

        elif interview_type == 'Resume':
            # Retrieve questions from the database
            from models.resume_upload import ResumeUpload
            resume_upload = db.session.query(ResumeUpload).filter_by(user_id=current_user.id).order_by(ResumeUpload.created_at.desc()).first()
            if not resume_upload or not resume_upload.custom_questions:
                flash("Please upload your resume to start Resume Mode.", "warning")
                return redirect(url_for('resume.upload'))
                
            resume_qs = json.loads(resume_upload.custom_questions)
            # Create temporary custom questions in the database
            for q_text in resume_qs:
                custom_q = Question(
                    text=q_text,
                    category='Resume',
                    difficulty='Medium',
                    expected_keywords=json.dumps(["experience", "project", "challenges", "skills"])
                )
                db.session.add(custom_q)
                db.session.commit()
                question_ids.append(custom_q.id)
            
        # Store in db state
        new_session.question_ids = json.dumps(question_ids)
        new_session.current_index = 0
        new_session.asked_follow_up = False
        new_session.last_question_id = None
        new_session.last_question_text = None
        db.session.commit()
        
        return redirect(url_for('interview.room', session_id=new_session.id))
        
    return render_template('select_interview.html')

@interview_bp.route('/interview/room/<int:session_id>')
@login_required
def room(session_id):
    """Renders the mock interview portal room."""
    interview_sess = db.get_or_404(InterviewSession, session_id)
    if interview_sess.user_id != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('interview.select'))
        
    if interview_sess.status == 'completed':
        flash("This interview has already been completed.", "info")
        return redirect(url_for('interview.feedback_page', session_id=session_id))
        
    q_ids = json.loads(interview_sess.question_ids or '[]')
    curr_idx = interview_sess.current_index
    
    if curr_idx >= len(q_ids) and not interview_sess.pending_follow_up:
        # Done! Finalize session
        return redirect(url_for('interview.finalize', session_id=session_id))
        
    # Determine what question to display
    is_follow_up = False
    
    if interview_sess.pending_follow_up:
        # Serve the AI-generated follow up question
        q_text = interview_sess.pending_follow_up
        q_id = None
        is_follow_up = True
    else:
        q_id = q_ids[curr_idx]
        question = db.session.get(Question, q_id)
        q_text = question.text
        interview_sess.last_question_id = q_id
        interview_sess.last_question_text = q_text
        db.session.commit()
        
    from models.skill_state import SessionSkillState
    skill_states_data = {}
    is_adaptive = interview_sess.experiment_mode != 'fixed'
    
    if is_adaptive:
        from services.adaptive_engine import AdaptiveEngine
        ae = AdaptiveEngine()
        session_skills = ae.get_skills_for_session(interview_sess)
        
        # Check if skill states already exist in DB
        states = SessionSkillState.query.filter_by(session_id=session_id).all()
        if not states:
            # Seed default 50.0 intermediate states for the session skills so they appear on start
            for skill in session_skills:
                state_seed = SessionSkillState(
                    session_id=session_id,
                    skill_name=skill,
                    score=50.0,
                    level='Intermediate'
                )
                db.session.add(state_seed)
            db.session.commit()
            states = SessionSkillState.query.filter_by(session_id=session_id).all()
            
        skill_states_data = {s.skill_name: {"score": s.score, "level": s.level} for s in states}

    return render_template(
        'interview.html',
        session_id=session_id,
        question_text=q_text,
        question_id=q_id,
        current_num=curr_idx + 1,
        total_num=len(q_ids),
        is_follow_up=is_follow_up,
        is_adaptive=is_adaptive,
        skill_states_json=json.dumps(skill_states_data)
    )

@interview_bp.route('/interview/submit_answer', methods=['POST'])
@login_required
def submit_answer():
    """
    Receives recorded answer files, transcribes speech, evaluates content,
    performs gaze estimation, saves response in database, and sets up 
    the next question or dynamic follow-ups.
    """
    stage = "Initialization"
    try:
        # Stage 1: Upload Started
        print("\n[ STAGE 1 ] Upload Started - Parsing form payload...")
        stage = "Upload Started"
        
        session_id = int(request.form.get('session_id'))
        duration = float(request.form.get('duration', 0.0))
        browser_transcript = request.form.get('browser_transcript', '')
        is_follow_up = request.form.get('is_follow_up') == 'true'
        question_text = request.form.get('question_text', '')
        
        # Load interview session from DB
        interview_sess = db.session.get(InterviewSession, session_id)
        if not interview_sess or interview_sess.user_id != current_user.id:
            print("[ ERROR ] Unauthorized session context validation failed.")
            return jsonify({"success": False, "stage": "Upload Started", "error": "Unauthorized session context"}), 403
            
        audio_file = request.files.get('audio')
        video_file = request.files.get('video')
        
        audio_path = None
        video_path = None
        
        # Setup session uploads folder
        session_upload_dir = Config.UPLOAD_FOLDER / f"session_{session_id}"
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Audio File if sent
        if audio_file and audio_file.filename != '':
            ext = secure_filename(audio_file.filename).split('.')[-1]
            filename = f"q_{interview_sess.current_index}_audio.{ext}"
            dest = session_upload_dir / filename
            audio_file.save(str(dest))
            audio_path = f"uploads/session_{session_id}/{filename}"
            print(f"[ INFO ] Saved audio clip to: {audio_path}")
            
        # Save Video File if sent
        if video_file and video_file.filename != '':
            ext = secure_filename(video_file.filename).split('.')[-1]
            filename = f"q_{interview_sess.current_index}_video.{ext}"
            dest = session_upload_dir / filename
            video_file.save(str(dest))
            video_path = f"uploads/session_{session_id}/{filename}"
            print(f"[ INFO ] Saved video clip to: {video_path}")
            
        # Stage 2: Upload Completed
        print("[ STAGE 2 ] Upload Completed - Response files successfully written to server.")
        stage = "Upload Completed"
        
        # Stage 3: Whisper Started
        print("[ STAGE 3 ] Whisper Started - Running speech transcription pipeline...")
        stage = "Whisper Started"
        
        transcript = "[Silence]"
        if audio_path:
            abs_audio_path = Config.BASE_DIR / audio_path
            try:
                transcript = whisper_service.transcribe(abs_audio_path, browser_transcript)
            except Exception as e:
                print(f"[ WARNING ] Whisper transcriber failed: {e}. Falling back to browser text.")
                transcript = browser_transcript or "[Unable to transcribe audio]"
        else:
            transcript = browser_transcript or "[No response text submitted]"
            
        print(f"[ INFO ] Speech Transcript text: \"{transcript}\"")
        
        # Stage 4: Whisper Completed & Analytics Started
        print("[ STAGE 4 ] Whisper Completed - Calculating speech WPM rate and filler frequency...")
        stage = "Whisper Completed"
        
        speech_analysis = whisper_service.analyze_speech(transcript, duration)
        
        # Run Gaze analysis (OpenCV + MediaPipe) - non-blocking fail-safe
        eye_contact_score = 80.0
        head_stability_score = 80.0
        attention_duration_score = 80.0
        confidence_score = 80.0
        
        if video_path:
            try:
                print("[ INFO ] Running OpenCV + MediaPipe eye contact check...")
                abs_video_path = Config.BASE_DIR / video_path
                cv_res = cv_service.analyze_video(abs_video_path)
                eye_contact_score = cv_res.get("eye_contact", 80.0)
                head_stability_score = cv_res.get("head_stability", 80.0)
                attention_duration_score = cv_res.get("attention_duration", 80.0)
                confidence_score = cv_res.get("confidence", 80.0)
                print(f"[ INFO ] Camera attention scores calculated: eye={eye_contact_score}, stability={head_stability_score}, duration={attention_duration_score}, conf={confidence_score}")
            except Exception as e:
                print(f"[ WARNING ] Computer vision gaze estimator failed: {e}. Using fallback defaults.")
                
        # Stage 5: Gemini Started
        print("[ STAGE 5 ] Gemini Started - Requesting answer evaluation ratings...")
        stage = "Gemini Started"
        
        q_id = interview_sess.last_question_id
        if not is_follow_up:
            q_ids = json.loads(interview_sess.question_ids or '[]')
            curr_idx = interview_sess.current_index
            if curr_idx < len(q_ids):
                q_id = q_ids[curr_idx]
                interview_sess.last_question_id = q_id
                db.session.commit()
                
        question = db.session.get(Question, q_id) if q_id else None
        keywords = question.expected_keywords if question else "[]"
        category = question.category if question else "General"
        
        try:
            if interview_sess.experiment_mode != 'fixed':
                eval_result = groq_service.evaluate_answer_adaptive(
                    question=question_text,
                    answer=transcript,
                    category=category,
                    skill=question.skill if question else None,
                    expected_keywords=keywords
                )
            else:
                eval_result = groq_service.evaluate_answer(
                    question=question_text,
                    answer=transcript,
                    category=category,
                    expected_keywords=keywords
                )
        except Exception as e:
            print(f"[ WARNING ] Groq grading failed: {e}. Executing rule-based fallback evaluations.")
            if interview_sess.experiment_mode != 'fixed':
                eval_result = groq_service._rule_based_evaluate_adaptive(
                    question_text, transcript, category, question.skill if question else None, keywords
                )
            else:
                eval_result = groq_service._rule_based_evaluate(question_text, transcript, category, keywords)
            
        # Stage 6: Groq Completed
        print("[ STAGE 6 ] Groq Completed - Content scores generated successfully.")
        stage = "Groq Completed"
        
        # Stage 7: Feedback Generated & Response Saved
        print("[ STAGE 7 ] Feedback Generated - Committing response records to SQLite...")
        stage = "Feedback Generated"
        
        new_response = Response(
            session_id=session_id,
            question_id=q_id,
            question_text=question_text,
            audio_path=audio_path,
            video_path=video_path,
            transcript=transcript,
            filler_count=speech_analysis["filler_count"],
            wpm=speech_analysis["wpm"],
            duration=duration,
            eye_contact_score=eye_contact_score,
            head_stability_score=head_stability_score,
            attention_duration_score=attention_duration_score,
            confidence_score=confidence_score,
            relevance_score=eval_result.get("relevance", 0.0),
            clarity_score=eval_result.get("clarity", eval_result.get("communication_quality", 0.0)),
            completeness_score=eval_result.get("completeness", 0.0),
            structure_score=eval_result.get("structure", 0.0),
            professionalism_score=eval_result.get("professionalism", 0.0),
            correctness_score=eval_result.get("correctness", 0.0),
            depth_score=eval_result.get("depth", 0.0),
            communication_quality_score=eval_result.get("communication_quality", 0.0),
            answer_score=eval_result.get("answer_score", 0.0),
            feedback=eval_result["feedback"]
        )
        new_response.set_filler_words(speech_analysis["filler_details"])
        
        # Update skill state if in adaptive mode
        if interview_sess.experiment_mode != 'fixed' and question and question.skill:
            db.session.add(new_response)
            db.session.flush()
            
            from services.adaptive_engine import AdaptiveEngine
            adaptive_engine = AdaptiveEngine()
            
            confidence_data = {
                "eye_contact_score": eye_contact_score,
                "filler_count": speech_analysis["filler_count"],
                "speech_confidence_score": eval_result.get("confidence", 8.0) * 10.0,
                "attention_duration_score": attention_duration_score,
                "head_stability_score": head_stability_score
            }
            
            adaptive_engine.update_skill_state(
                session_id=session_id,
                skill_name=question.skill,
                score=new_response.answer_score,
                response_id=new_response.id,
                confidence_data=confidence_data
            )
        
        # Check dynamic AI follow up (only for fixed mode)
        follow_up_generated = None
        asked_follow_up = interview_sess.asked_follow_up
        curr_idx = interview_sess.current_index
        q_ids = json.loads(interview_sess.question_ids or '[]')
        
        should_follow_up = (not is_follow_up) and (not asked_follow_up) and (curr_idx in [1, 3]) and (speech_analysis["word_count"] > 15) and (curr_idx < len(q_ids) - 1) and (interview_sess.experiment_mode == 'fixed')
        
        if should_follow_up:
            try:
                print("[ INFO ] Spawning dynamic follow-up question via Groq...")
                follow_up_generated = groq_service.generate_follow_up(question_text, transcript)
                interview_sess.pending_follow_up = follow_up_generated
                interview_sess.asked_follow_up = True
                new_response.follow_up_question = follow_up_generated
                print(f"[ INFO ] Dynamic Follow-Up Question: \"{follow_up_generated}\"")
            except Exception as e:
                print(f"[ WARNING ] Follow-up generation failed: {e}. Skipping follow-up stage.")
                
        db.session.add(new_response)
        try:
            from models.interview_response import InterviewResponse
            new_interview_response = InterviewResponse(
                interview_id=session_id,
                question=question_text,
                answer=transcript,
                skill=question.skill if (question and question.skill) else "General",
                score=eval_result.get("answer_score", 0.0)
            )
            db.session.add(new_interview_response)
        except Exception as e_resp:
            print(f"[ WARNING ] Mirroring to InterviewResponse failed: {e_resp}")
        db.session.commit()
        
        # Advance index if we didn't just spawn a follow up
        if interview_sess.pending_follow_up and not is_follow_up:
            pass
        else:
            if is_follow_up:
                interview_sess.pending_follow_up = None
            interview_sess.current_index = curr_idx + 1
            
        db.session.commit()
        
        next_idx = interview_sess.current_index
        
        # If in adaptive mode and under 5 questions, dynamically choose and append the next question
        if interview_sess.experiment_mode != 'fixed' and next_idx < 5:
            from services.adaptive_engine import AdaptiveEngine
            adaptive_engine = AdaptiveEngine()
            next_q, select_reason = adaptive_engine.get_next_question(interview_sess, new_response)
            if next_q:
                q_ids.append(next_q.id)
                interview_sess.question_ids = json.dumps(q_ids)
                db.session.commit()
                
        total_qs = len(q_ids)
        is_done = next_idx >= total_qs and not interview_sess.pending_follow_up
        
        # Calculate B.Tech research live dashboard values
        from models.skill_state import SessionSkillState
        from services.adaptive_engine import AdaptiveEngine
        adaptive_engine = AdaptiveEngine()
        
        comp_map = adaptive_engine.detect_competency_boundaries(session_id)
        interview_sess.competency_map = json.dumps(comp_map)
        db.session.commit()
        
        states = SessionSkillState.query.filter_by(session_id=session_id).all()
        states_data = {s.skill_name: {"score": s.score, "level": s.level} for s in states}
        
        avg_score = sum(s.score for s in states) / len(states) if states else 50.0
        overall_level = "Beginner" if avg_score < 50.0 else ("Intermediate" if avg_score < 80.0 else "Advanced")
        
        dec_log = interview_sess.get_decision_log()
        next_difficulty = "Medium"
        if dec_log:
            next_difficulty = dec_log[-1].get("target_difficulty", "Medium")
            
        weak_skills = [s.skill_name for s in states if s.score < 50.0]
        
        # Determine next question details to serve dynamically (prevent page reload)
        next_q_text = ""
        next_q_id = None
        is_follow_up_next = False
        
        if interview_sess.pending_follow_up:
            next_q_text = interview_sess.pending_follow_up
            is_follow_up_next = True
        elif next_idx < len(q_ids):
            next_q_id = q_ids[next_idx]
            nq = db.session.get(Question, next_q_id)
            next_q_text = nq.text if nq else ""
            is_follow_up_next = False
            
            # Sync last_question fields
            interview_sess.last_question_id = next_q_id
            interview_sess.last_question_text = next_q_text
            db.session.commit()

        print("[ SUCCESS ] Response saved and committed successfully.")
        return jsonify({
            "success": True,
            "is_done": is_done,
            "next_num": next_idx + 1 if not interview_sess.pending_follow_up else next_idx + 2,
            "total_num": total_qs,
            "has_follow_up": bool(interview_sess.pending_follow_up),
            "is_adaptive": interview_sess.experiment_mode != 'fixed',
            "skill_states": states_data,
            "overall_level": overall_level,
            "current_difficulty": next_difficulty,
            "completion_percentage": round((min(next_idx, total_qs) / max(total_qs, 1)) * 100, 1),
            "weak_skills": weak_skills,
            "next_question_text": next_q_text,
            "next_question_id": next_q_id,
            "is_follow_up_next": is_follow_up_next
        })
        
    except Exception as e:
        print(f"[ CRITICAL ] Exception during {stage} stage: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "stage": stage, "error": str(e)}), 500

@interview_bp.route('/interview/finalize/<int:session_id>')
@login_required
def finalize(session_id):
    """Aggregates all question grades, runs Gemini session review, and commits feedback summaries."""
    interview_sess = db.get_or_404(InterviewSession, session_id)
    
    if interview_sess.status == 'completed':
        return redirect(url_for('interview.feedback_page', session_id=session_id))
        
    responses = db.session.query(Response).filter_by(session_id=session_id).all()
    if not responses:
        flash("No responses recorded.", "warning")
        return redirect(url_for('interview.select'))
        
    # Compute Averages
    count = len(responses)
    avg_relevance = sum(r.relevance_score for r in responses) / count
    avg_clarity = sum(r.clarity_score for r in responses) / count
    avg_completeness = sum(r.completeness_score for r in responses) / count
    avg_structure = sum(r.structure_score for r in responses) / count
    avg_professionalism = sum(r.professionalism_score for r in responses) / count
    avg_eye_contact = sum(r.eye_contact_score for r in responses) / count
    
    # Analyze Fillers and WPM to compute Communication Score
    total_fillers = sum(r.filler_count for r in responses)
    total_words = sum(len(r.transcript.split()) for r in responses)
    filler_ratio = (total_fillers / total_words) * 100 if total_words > 0 else 0.0
    
    if interview_sess.experiment_mode != 'fixed':
        from models.skill_state import SessionSkillState
        states = SessionSkillState.query.filter_by(session_id=session_id).all()
        if states:
            tech_score = sum(s.score for s in states) / len(states)
        else:
            tech_score = (sum(r.correctness_score for r in responses) / count) * 10.0
            
        avg_comm_qual = sum(r.communication_quality_score for r in responses) / count
        comm_score = avg_comm_qual * 10.0
        
        avg_correctness = sum(r.correctness_score for r in responses) / count
        avg_depth = sum(r.depth_score for r in responses) / count
        quality_score = ((avg_correctness + avg_depth + avg_relevance + avg_completeness) / 4.0) * 10.0
        
        prof_score = avg_professionalism * 10.0
        
        fluency_score = max(30.0, 100.0 - (filler_ratio * 8.0))
        conf_score = (avg_eye_contact * 0.6) + (fluency_score * 0.4)
        
        # Ensure values stay in 30-100 range
        tech_score = max(30.0, min(100.0, tech_score))
        comm_score = max(30.0, min(100.0, comm_score))
        quality_score = max(30.0, min(100.0, quality_score))
        prof_score = max(30.0, min(100.0, prof_score))
        conf_score = max(30.0, min(100.0, conf_score))
        
        overall_score = (comm_score * 0.2) + (tech_score * 0.3) + (quality_score * 0.25) + (prof_score * 0.15) + (conf_score * 0.1)
        
        from services.adaptive_engine import AdaptiveEngine
        adaptive_engine = AdaptiveEngine()
        comp_map = adaptive_engine.detect_competency_boundaries(session_id)
        interview_sess.competency_map = json.dumps(comp_map)
    else:
        # Communication score: Based on clarity, filler words deduction, and WPM check
        # Max deductions for high fillers
        filler_deduction = min(3.0, filler_ratio * 0.5)
        comm_score = (avg_clarity * 10.0) - (filler_deduction * 10.0)
        comm_score = max(30.0, min(100.0, comm_score))
        
        # Technical Knowledge Score
        tech_score = avg_relevance * 10.0
        tech_score = max(30.0, min(100.0, tech_score))
        
        # Answer Quality
        quality_score = ((avg_relevance + avg_completeness + avg_structure) / 3.0) * 10.0
        quality_score = max(30.0, min(100.0, quality_score))
        
        # Professionalism
        prof_score = avg_professionalism * 10.0
        prof_score = max(30.0, min(100.0, prof_score))
        
        # Confidence Score: 60% Gaze attention + 40% fluency (absence of excessive fillers)
        fluency_score = max(30.0, 100.0 - (filler_ratio * 8.0))
        conf_score = (avg_eye_contact * 0.6) + (fluency_score * 0.4)
        conf_score = max(30.0, min(100.0, conf_score))
        
        # Overall Score (Weighted Average)
        overall_score = (comm_score * 0.2) + (tech_score * 0.3) + (quality_score * 0.25) + (prof_score * 0.15) + (conf_score * 0.1)
    
    # Aggregated fields
    questions_list = [r.question_text for r in responses]
    answers_list = [r.transcript or "" for r in responses]
    combined_transcript = "\n\n".join([f"Q: {r.question_text}\nA: {r.transcript or ''}" for r in responses])
    total_duration = sum(r.duration for r in responses)
    avg_eye_contact = sum(r.eye_contact_score for r in responses) / count
    total_fillers = sum(r.filler_count for r in responses)

    # Store scores and aggregated metadata
    interview_sess.role_applied = interview_sess.role_applied or "Software Engineer"
    interview_sess.questions_asked = json.dumps(questions_list)
    interview_sess.user_answers = json.dumps(answers_list)
    interview_sess.audio_transcript = combined_transcript
    interview_sess.duration = int(total_duration)
    interview_sess.eye_contact_score = round(avg_eye_contact, 1)
    interview_sess.filler_word_count = total_fillers
    interview_sess.communication_score = round(comm_score, 1)
    interview_sess.technical_score = round(tech_score, 1)
    interview_sess.answer_quality_score = round(quality_score, 1)
    interview_sess.professionalism_score = round(prof_score, 1)
    interview_sess.confidence_score = round(conf_score, 1)
    interview_sess.overall_score = round(overall_score, 1)
    
    if interview_sess.interview_type == 'Project':
        category_scores = {}
        from models.skill_state import SessionSkillState
        states = {s.skill_name: s.score for s in SessionSkillState.query.filter_by(session_id=session_id).all()}
        
        response_scores = {}
        for r in responses:
            q = db.session.get(Question, r.question_id) if r.question_id else None
            if q and q.skill:
                if q.skill not in response_scores:
                    response_scores[q.skill] = []
                response_scores[q.skill].append(r.answer_score)
        
        categories_list = [
            "Project Understanding",
            "Architecture",
            "Technology Stack",
            "Implementation",
            "Challenges",
            "Scalability",
            "Contribution Verification"
        ]
        
        for cat in categories_list:
            if cat in states:
                category_scores[cat] = states[cat]
            elif cat in response_scores:
                category_scores[cat] = sum(response_scores[cat]) / len(response_scores[cat])
            else:
                category_scores[cat] = 50.0
                
        project_scores = {
            "Project Understanding Score": round(category_scores.get("Project Understanding", 50.0), 1),
            "Architecture Score": round(category_scores.get("Architecture", 50.0), 1),
            "Technology Knowledge Score": round(category_scores.get("Technology Stack", 50.0), 1),
            "Implementation Depth Score": round(category_scores.get("Implementation", 50.0), 1),
            "Problem Solving Score": round((category_scores.get("Challenges", 50.0) + category_scores.get("Scalability", 50.0)) / 2.0, 1),
            "Communication Score": round(comm_score, 1),
            "Confidence Score": round(conf_score, 1)
        }
        
        project_feedback = {
            "scores": project_scores,
            "strengths": [],
            "weaknesses": [],
            "improvements": [],
            "revision_topics": [],
            "presentation_tips": []
        }
        
        if groq_service.initialized:
            try:
                transcript_summary = ""
                for idx, r in enumerate(responses):
                    transcript_summary += f"Q: {r.question_text}\nA: {r.transcript or ''}\n\n"
                    
                prompt = f"""
You are a senior engineering hiring manager evaluating a candidate's Project Defense interview.
Here is the actual transcript of the interview questions and candidate answers:

{transcript_summary}

Provide a structured, rigorous assessment of their project presentation and technical depth.
Evaluate with real engineering standards:
- Did they demonstrate deep architectural understanding or just repeat high-level bullet points?
- Were their explanations of technology choices grounded in engineering trade-offs?
- Did they clearly prove individual contribution vs boilerplate code?

Respond strictly with a JSON object containing:
{{
  "strengths": ["3 specific engineering strengths demonstrated in their answers"],
  "weaknesses": ["2–3 specific technical gaps, vague explanations, or omitted mechanisms"],
  "improvements": ["2–3 actionable technical steps to improve the project's codebase or design"],
  "revision_topics": ["Specific technologies, architectural patterns, or algorithms they should revise"],
  "presentation_tips": ["Actionable coaching on how to present system architectures and trade-offs more convincingly in real interviews"]
}}
"""
                llm_res = groq_service._create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=True,
                    temperature=0.2,
                    max_tokens=800
                )
                llm_text = llm_res.choices[0].message.content.strip()
                if llm_text.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_text)
                    if match:
                        llm_text = match.group(1).strip()
                parsed_feedback = json.loads(llm_text)
                project_feedback.update(parsed_feedback)
            except Exception as ex:
                print(f"[ WARNING ] Failed to generate project feedback: {ex}")
                
        if not project_feedback.get("strengths"):
            project_feedback["strengths"] = [
                "Demonstrated basic understanding of project features.",
                "Mentioned key technologies used in implementation."
            ]
            project_feedback["weaknesses"] = [
                "Could provide deeper details on system architecture decisions.",
                "Could clarify direct personal contribution vs template code."
            ]
            project_feedback["improvements"] = [
                "Incorporate comprehensive logging and error handling.",
                "Create a clear architectural diagram representing data flow."
            ]
            project_feedback["revision_topics"] = [
                "Database design and normalization rules.",
                "API design guidelines and authentication protocol flow."
            ]
            project_feedback["presentation_tips"] = [
                "Use the STAR method to describe how you tackled engineering challenges.",
                "Be ready to explain why you chose your specific technology stack over alternatives."
            ]
            
        try:
            curr_map = json.loads(interview_sess.competency_map or '{}')
        except Exception:
            curr_map = {}
        curr_map["project_feedback"] = project_feedback
        interview_sess.competency_map = json.dumps(curr_map)
        db.session.commit()
    
    # 6. Generate Explainable Feedback Summary & Suggestions via Groq
    if groq_service.initialized:
        try:
            # Build concise transcript digest of actual candidate responses
            responses_digest = ""
            for idx, r in enumerate(responses):
                ans_text = (r.transcript or "").strip() or "[No audible response recorded]"
                q_text = r.question_text
                score_val = r.answer_score if r.answer_score is not None else 0.0
                responses_digest += f"Q{idx+1}: {q_text}\nCandidate Answer: {ans_text[:350]}\nScore: {score_val:.1f}/100\n\n"

            prompt = f"""
You are a senior engineering hiring manager and executive interviewer conducting a comprehensive, objective assessment of a candidate who completed a {interview_sess.interview_type} mock interview for the role of '{interview_sess.role_applied}'.

Diagnostic Performance Metrics:
- Overall Score: {overall_score:.1f}/100
- Technical Competency: {tech_score:.1f}/100
- Answer Quality & Depth: {quality_score:.1f}/100
- Communication & Cohesion: {comm_score:.1f}/100
- Professionalism & Engineering Vocabulary: {prof_score:.1f}/100
- Behavioral Signals: Average Eye Contact {avg_eye_contact:.1f}%, Speech Pace {avg_wpm:.0f} WPM, Filler Words: {total_fillers}

Interview Transcript Digest:
{responses_digest}

Instructions for the Executive Report:
1. "feedback_summary": Write a thorough, structured, executive-level technical assessment (2–3 paragraphs).
   - Verdict & Placement Readiness: Clear assessment of placement readiness based on the overall score without generic platitudes or unearned praise.
   - Demonstrated Strengths: Explicitly reference specific questions/topics where the candidate demonstrated solid technical grasp.
   - Deficiencies & Gaps: Call out the weakest answers, omitted mechanisms (complexity, memory trade-offs, internal architecture), or misconceptions.
   - Delivery & Communication: Synthesize speech pace, filler frequency, and articulation clarity.
   - Do NOT invent candidate claims or strengths. Base every observation on their actual answers above.
2. "recommendations": Provide exactly 5 prioritized, highly specific, actionable engineering recommendations.
   - Must address their demonstrated weak areas (e.g., "Deepen knowledge of B-Tree indexing vs hash indexing", "Practice structuring responses with the STAR method").
   - Format each recommendation as a single clear sentence.

Format your response strictly as a JSON object:
{{
  "feedback_summary": "Thorough, structured recruiter evaluation paragraph(s)...",
  "recommendations": [
    "Prioritized actionable recommendation 1...",
    "Prioritized actionable recommendation 2...",
    "Prioritized actionable recommendation 3...",
    "Prioritized actionable recommendation 4...",
    "Prioritized actionable recommendation 5..."
  ]
}}
"""
            
            response = groq_service._create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.25,
                max_tokens=1400
            )
            resp_text = response.choices[0].message.content.strip()
            
            if resp_text.startswith("```"):
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", resp_text)
                if match:
                    resp_text = match.group(1).strip()
                    
            report = json.loads(resp_text)
            interview_sess.feedback_summary = report.get("feedback_summary")
            recs = report.get("recommendations")
            if isinstance(recs, list):
                recs_str = []
                for i, r in enumerate(recs):
                    r_stripped = str(r).strip()
                    if re.match(r'^\d+[\s\.\-\:]', r_stripped):
                        recs_str.append(r_stripped)
                    else:
                        recs_str.append(f"{i+1}. {r_stripped}")
                interview_sess.recommendations = "\n".join(recs_str)
            else:
                interview_sess.recommendations = str(recs) if recs is not None else ""
        except Exception as e:
            print(f"Failed to generate session feedback via Groq: {e}")
            
    # Fallback feedback if Gemini fails or is not setup
    if not interview_sess.feedback_summary:
        # Generate clean heuristics-based feedback
        recs = []
        feedback_points = []
        
        if comm_score < 75:
            feedback_points.append("Your communication score was reduced because you relied on filler words and paused frequently.")
            recs.append("Practice speaking in front of a mirror or recording yourself to reduce filler words (uh, like, you know).")
        else:
            feedback_points.append("You showed strong communication and articulation skills throughout the session.")
            recs.append("Maintain your current speed and continue practicing structured speaking.")
            
        if tech_score < 75:
            feedback_points.append("Your technical score indicates that key concepts or details were missing from technical explanations.")
            recs.append("Review foundational data structures and algorithms, focusing on time/space complexity analysis.")
        else:
            feedback_points.append("Your answers demonstrated solid technical grasp and reference to critical domain keywords.")
            recs.append("Incorporate more real-world software architecture examples into your technical explanations.")
            
        if avg_eye_contact < 75:
            feedback_points.append("Your confidence score was lowered because eye contact consistency was inconsistent.")
            recs.append("Practice looking directly at the webcam instead of the screen or keyboard during online assessments.")
        else:
            recs.append("Continue to maintain strong posture and active focus during video calls.")
            
        recs.append("Structure all future behavioral answers using the STAR (Situation, Task, Action, Result) methodology.")
        recs.append("Take 5 seconds to organize your thoughts before speaking to avoid pacing issues.")
        
        interview_sess.feedback_summary = " ".join(feedback_points) + " Overall, this mock interview highlights your strengths and key areas that require improvement before your placements."
        interview_sess.recommendations = "\n".join([f"{i+1}. {r}" for i, r in enumerate(recs)])
        
    interview_sess.status = 'completed'
    db.session.commit()
    
    # Save PerformanceReport in database
    try:
        from models.performance_report import PerformanceReport
        db.session.query(PerformanceReport).filter_by(session_id=session_id).delete()
        perf_report = PerformanceReport(
            user_id=interview_sess.user_id,
            session_id=interview_sess.id,
            technical_score=interview_sess.technical_score,
            communication_score=interview_sess.communication_score,
            confidence_score=interview_sess.confidence_score,
            overall_score=interview_sess.overall_score,
            feedback_summary=interview_sess.feedback_summary
        )
        db.session.add(perf_report)
        db.session.commit()
    except Exception as ex:
        print(f"Error logging PerformanceReport to database: {ex}")
        
    # Recalculate user details and trigger Admin Notifications
    try:
        from models.user import User
        user = db.session.query(User).get(interview_sess.user_id)
        if user:
            completed_count = db.session.query(InterviewSession).filter_by(user_id=user.id, status='completed').count()
            user.total_interviews = completed_count
            
            avg_overall = db.session.query(db.func.avg(InterviewSession.overall_score)).filter_by(user_id=user.id, status='completed').scalar() or 0.0
            user.average_score = round(float(avg_overall), 1)
            
            max_overall = db.session.query(db.func.max(InterviewSession.overall_score)).filter_by(user_id=user.id, status='completed').scalar() or 0.0
            user.highest_score = round(float(max_overall), 1)
            
            if user.average_score < 50:
                user.skill_level = 'Beginner'
            elif user.average_score < 80:
                user.skill_level = 'Intermediate'
            else:
                user.skill_level = 'Advanced'
                
            total_time_secs = db.session.query(db.func.sum(InterviewSession.duration)).filter_by(user_id=user.id).scalar() or 0
            user.total_practice_time = int(total_time_secs)
            
            db.session.commit()
            
            # Extract improvement areas
            improvements = []
            if interview_sess.communication_score < 75:
                improvements.append("Reduce filler words and improve vocal pace.")
            if interview_sess.technical_score < 75:
                improvements.append("Strengthen conceptual data structures & algorithms knowledge.")
            if interview_sess.eye_contact_score < 75:
                improvements.append("Improve eye contact consistency by looking at the camera.")
            interview_sess.improvement_areas = "\n".join(improvements) if improvements else "None. Strong performance."
            db.session.commit()
            
            # Trigger Admin Notifications
            from models.admin_notification import AdminNotification
            notif_comp = AdminNotification(
                type='interview_completed',
                message=f"User '{user.username}' completed a {interview_sess.interview_type} mock interview with score {interview_sess.overall_score}%"
            )
            db.session.add(notif_comp)
            
            if interview_sess.overall_score >= 90:
                notif_high = AdminNotification(
                    type='high_score',
                    message=f"High score alert! User '{user.username}' achieved a score of {interview_sess.overall_score}% in a {interview_sess.interview_type} interview"
                )
                db.session.add(notif_high)
            db.session.commit()
            
            # Trigger User Email Notification
            if user.interview_reports_enabled:
                strengths_list = []
                if interview_sess.communication_score >= 75:
                    strengths_list.append("Strong verbal pace and professional tone.")
                if interview_sess.technical_score >= 75:
                    strengths_list.append("Good technical domain knowledge.")
                if interview_sess.confidence_score >= 75:
                    strengths_list.append("High confidence and consistent webcam eye contact.")
                if not strengths_list:
                    strengths_list.append("Completed all questions within duration guidelines.")
                    
                weaknesses_list = [w.strip() for w in getattr(interview_sess, 'improvement_areas', '').split('\n') if w.strip()]
                
                recs_list = [r.strip() for r in (interview_sess.recommendations or '').split('\n') if r.strip()]
                cleaned_recs = []
                for r in recs_list:
                    cleaned = re.sub(r'^\d+[\.\)\s\-]+', '', r)
                    cleaned_recs.append(cleaned)
                    
                feedback_url = url_for('interview.feedback_page', session_id=interview_sess.id, _external=True)
                
                EmailService.send_email_async(
                    to_email=user.email,
                    subject="Mock Interview Performance Report - HireWise AI",
                    template_name="interview_report.html",
                    context={
                        "name": user.name,
                        "overall_score": interview_sess.overall_score,
                        "interview_type": interview_sess.interview_type,
                        "company_name": interview_sess.company_name,
                        "technical_score": interview_sess.technical_score,
                        "communication_score": interview_sess.communication_score,
                        "confidence_score": interview_sess.confidence_score,
                        "feedback_summary": interview_sess.feedback_summary,
                        "strengths": strengths_list,
                        "weaknesses": weaknesses_list,
                        "recommendations": cleaned_recs,
                        "feedback_url": feedback_url
                    },
                    user_id=user.id,
                    event_type="interview_report",
                    ip_address=request.remote_addr
                )
                
            # Trigger Admin Email Notification
            from flask import current_app
            admin_url = url_for('admin.dashboard', _external=True)
            EmailService.send_email_async(
                to_email=current_app.config.get('ADMIN_EMAIL', 'admin@hirewise.ai'),
                subject="Admin Alert: Interview Completed",
                template_name="admin_notification.html",
                context={
                    "notification_type": "interview_completed",
                    "message": f"User '{user.username}' completed a {interview_sess.interview_type} mock interview with score {interview_sess.overall_score}%",
                    "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    "username": user.username,
                    "extra_details": {
                        "Interview Type": interview_sess.interview_type,
                        "Overall Score": f"{interview_sess.overall_score}%",
                        "Technical Score": f"{interview_sess.technical_score}%",
                        "Communication Score": f"{interview_sess.communication_score}%"
                    },
                    "admin_url": admin_url
                },
                event_type="admin_notification",
                ip_address=request.remote_addr
            )
    except Exception as ex:
        print(f"Error updating user scorecard statistics: {ex}")
    
    # Progression state is kept in DB but we reset or finalize session context here
    pass
    
    return redirect(url_for('interview.feedback_page', session_id=session_id))

@interview_bp.route('/interview/feedback/<int:session_id>')
@login_required
def feedback_page(session_id):
    """Displays the comprehensive scorecard analysis dashboard for a completed session."""
    interview_sess = db.get_or_404(InterviewSession, session_id)
    
    if interview_sess.user_id != current_user.id:
        flash("Access Denied.", "danger")
        return redirect(url_for('dashboard.home'))
        
    responses = db.session.query(Response).filter_by(session_id=session_id).all()
    
    # Split recommendations string into a list for clean template rendering
    recommendation_list = []
    if interview_sess.recommendations:
        # Split by newline and clean empty items
        recommendation_list = [r.strip() for r in interview_sess.recommendations.split('\n') if r.strip()]
        # Strip numbering prefix (e.g. "1. ") if present
        cleaned_recs = []
        for r in recommendation_list:
            cleaned = re.sub(r'^\d+[\.\)\s\-]+', '', r)
            cleaned_recs.append(cleaned)
        recommendation_list = cleaned_recs
        
    # Gauge and chart styling variables mapping
    return render_template(
        'feedback.html',
        session=interview_sess,
        responses=responses,
        recommendations=recommendation_list
    )
