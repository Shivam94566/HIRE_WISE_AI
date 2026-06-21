import os
import json
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from config import Config
from services.groq_service import GroqService
from services.whisper_service import WhisperService
from services.resume_service import ResumeService
from services.email_service import EmailService

resume_bp = Blueprint('resume', __name__)

# Initialize Services
resume_service = ResumeService()
groq_service = GroqService()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf'}

@resume_bp.route('/resume/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Renders the resume upload page and processes uploaded resume PDFs."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'resume' not in request.files:
            flash('No file part in the request.', 'danger')
            return redirect(request.url)
            
        file = request.files['resume']
        
        # If user does not select file, browser also submits an empty part
        if file.filename == '':
            flash('No selected file. Please choose a PDF resume.', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Make a unique filename for the user using current_user
            unique_filename = f"user_{current_user.id}_resume_{filename}"
            save_path = Config.UPLOAD_FOLDER / unique_filename
            
            try:
                # Save file
                file.save(str(save_path))
                
                # Extract text using PyPDF2 resume service
                extracted_text = resume_service.extract_text(save_path)
                
                if not extracted_text or len(extracted_text.strip()) < 50:
                    flash('Could not extract sufficient text from your PDF. Please ensure it is not a scanned image.', 'warning')
                    return redirect(request.url)
                
                # Generate custom questions based on resume content via Groq
                try:
                    questions = groq_service.generate_resume_questions(extracted_text)
                except Exception as e:
                    print(f"Error generating questions via Groq: {e}")
                    questions = []
                    
                if not questions or len(questions) < 5:
                    questions = [
                        "Describe a major technical project you worked on. What was your role and the outcome?",
                        "What programming languages and tools from your resume are you most comfortable with, and why?",
                        "Tell me about a challenging bug or problem you solved in one of your projects.",
                        "How do you keep your technical skills up to date with the latest industry trends?",
                        "Explain a situation where you had to learn a new tool or technology quickly for a project."
                    ]
                    
                # Save Resume Metadata and custom questions in database
                try:
                    from database.connection import db
                    from models.resume_upload import ResumeUpload
                    
                    # Clean up old upload entries for this user
                    db.session.query(ResumeUpload).filter_by(user_id=current_user.id).delete()
                    
                    # Heuristically parse skills from text
                    known_skills = ['python', 'javascript', 'html', 'css', 'react', 'node', 'sql', 'sqlite', 'mongodb', 'c++', 'java', 'opencv', 'mediapipe', 'whisper', 'flask', 'django', 'machine learning', 'scikit-learn']
                    skills_found = []
                    text_lower = extracted_text.lower()
                    for skill in known_skills:
                        if skill in text_lower:
                            skills_found.append(skill.title())
                    skills_str = ", ".join(skills_found) if skills_found else "Software Engineering"
                    
                    # Deep analysis via Groq Service
                    try:
                        analysis = groq_service.analyze_resume_details(extracted_text)
                    except Exception as e:
                        print(f"Error in deep resume parsing: {e}")
                        analysis = {}
                        
                    upload_log = ResumeUpload(
                        user_id=current_user.id,
                        filename=filename,
                        file_path=str(save_path),
                        parsed_text=extracted_text,
                        skills_extracted=analysis.get("skills", skills_str),
                        projects=analysis.get("projects", "No projects extracted"),
                        experience=analysis.get("experience", "No experience extracted"),
                        certifications=analysis.get("certifications", "No certifications extracted"),
                        missing_skills=analysis.get("missing_skills", "No missing skills identified"),
                        ats_score=analysis.get("ats_score", 60),
                        suggestions_generated=analysis.get("suggestions", "No suggestions generated"),
                        custom_questions=json.dumps(questions)
                    )
                    db.session.add(upload_log)
                    try:
                        from models.resume import Resume
                        new_resume = Resume(
                            user_id=current_user.id,
                            file_path=str(save_path),
                            extracted_skills=analysis.get("skills", skills_str),
                            projects=analysis.get("projects", "No projects extracted"),
                            ats_score=analysis.get("ats_score", 60),
                            uploaded_at=datetime.utcnow()
                        )
                        db.session.add(new_resume)
                    except Exception as e_res:
                        print(f"[ WARNING ] Mirroring to Resume failed: {e_res}")
                    db.session.commit()
                    
                    # Trigger Admin Notification
                    from models.admin_notification import AdminNotification
                    notif = AdminNotification(
                        type='resume_upload',
                        message=f"Resume uploaded by user '{current_user.username}': ATS Score {upload_log.ats_score}%"
                    )
                    db.session.add(notif)
                    db.session.commit()
                    
                    # Trigger User Email Notification
                    if current_user.resume_notifications_enabled:
                        skills_list = [s.strip() for s in upload_log.skills_extracted.split(',') if s.strip()]
                        skills_count = len(skills_list)
                        
                        projects_list = [p.strip() for p in upload_log.projects.split('\n') if p.strip()]
                        projects_count = len(projects_list)
                        
                        interview_url = url_for('interview.select', _external=True)
                        
                        EmailService.send_email_async(
                            to_email=current_user.email,
                            subject="Resume Successfully Uploaded - HireWise AI",
                            template_name="resume_uploaded.html",
                            context={
                                "name": current_user.name,
                                "ats_score": upload_log.ats_score,
                                "upload_date": datetime.utcnow().strftime('%B %d, %Y'),
                                "filename": filename,
                                "skills_count": skills_count,
                                "projects_count": projects_count,
                                "skills_extracted": upload_log.skills_extracted,
                                "interview_url": interview_url
                            },
                            user_id=current_user.id,
                            event_type="resume_upload",
                            ip_address=request.remote_addr
                        )
                        
                    # Trigger Admin Email Notification
                    from flask import current_app
                    admin_url = url_for('admin.dashboard', _external=True)
                    EmailService.send_email_async(
                        to_email=current_app.config.get('ADMIN_EMAIL', 'admin@hirewise.ai'),
                        subject="Admin Alert: Resume Uploaded",
                        template_name="admin_notification.html",
                        context={
                            "notification_type": "resume_upload",
                            "message": f"Resume uploaded by user '{current_user.username}': ATS Score {upload_log.ats_score}%",
                            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                            "username": current_user.username,
                            "extra_details": {
                                "ATS Score": f"{upload_log.ats_score}%",
                                "Filename": filename
                            },
                            "admin_url": admin_url
                        },
                        event_type="admin_notification",
                        ip_address=request.remote_addr
                    )
                except Exception as ex:
                    print(f"Error logging resume database upload: {ex}")
                    db.session.rollback()
                    
                flash('Resume uploaded and parsed successfully! Ready to start your customized mock interview.', 'success')
                
                # Auto-redirect to interview selection with Resume Mode unlocked
                return render_template('select_interview.html', resume_unlocked=True)
                
            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'danger')
                print(f"Error handling PDF upload: {e}")
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload a PDF file.', 'danger')
            return redirect(request.url)
            
    return render_template('resume_upload.html')
