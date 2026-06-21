import os
import glob
import re
import uuid
import json
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from config import Config
from database.connection import db
from models.chat import ChatSession, ChatMessage, PracticeHistory
from services.mentor_service import MentorService, LAST_DEBUG_INFO
from services.whisper_service import WhisperService
from services.resume_service import ResumeService

mentor_bp = Blueprint('mentor', __name__)

# Instantiate Services
mentor_service = MentorService()
whisper_service = WhisperService()
resume_service = ResumeService()

@mentor_bp.route('/mentor')
@login_required
def chat():
    """Renders the main conversational mentor workspace and loads chat history."""
    sessions = db.session.query(ChatSession).filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    
    # Active session detection
    active_session_id = request.args.get('session_id', type=int)
    active_session = None
    
    if active_session_id:
        active_session = db.session.query(ChatSession).filter_by(id=active_session_id, user_id=current_user.id).first()
    
    if not active_session and sessions:
        active_session = sessions[0]
        
    messages = []
    if active_session:
        messages = db.session.query(ChatMessage).filter_by(session_id=active_session.id).order_by(ChatMessage.created_at.asc()).all()
        
    # Check if a resume is already uploaded for personalization
    user_resume_pattern = os.path.join(Config.UPLOAD_FOLDER, f"user_{current_user.id}_resume_*.pdf")
    has_resume = len(glob.glob(user_resume_pattern)) > 0
    
    # Fetch practice stats for display
    practice_count = db.session.query(PracticeHistory).filter_by(user_id=current_user.id).count()
    avg_practice_score = db.session.query(db.func.avg(PracticeHistory.score)).filter(PracticeHistory.user_id == current_user.id).scalar() or 0.0
    
    # Load recent topics
    recent_practices = db.session.query(PracticeHistory).filter_by(user_id=current_user.id).order_by(PracticeHistory.created_at.desc()).limit(5).all()
    
    return render_template(
        'mentor_chat.html',
        sessions=sessions,
        active_session=active_session,
        messages=messages,
        has_resume=has_resume,
        practice_count=practice_count,
        avg_score=round(avg_practice_score, 1),
        recent_practices=recent_practices
    )

@mentor_bp.route('/mentor/session/new', methods=['POST'])
@login_required
def new_session():
    """Creates a new conversational chat session with a tailored greeting based on selected mode."""
    mode = request.form.get('mode', 'chat')
    title = request.form.get('title', 'New Mentor Chat')
    
    # Limit title size
    if len(title) > 80:
        title = title[:80] + "..."
        
    new_sess = ChatSession(
        user_id=current_user.id,
        mode=mode,
        title=title
    )
    db.session.add(new_sess)
    db.session.commit()
    
    # Update user mentor sessions count
    current_user.mentor_sessions_count = (current_user.mentor_sessions_count or 0) + 1
    db.session.commit()
    
    # Initial greeting based on mode
    greetings = {
        "chat": "Hello! I am your HireWise Career Mentor. I'm here to answer your questions about placement preparation, resumes, career strategies, or coding interviews. What would you like to discuss today?",
        "practice_hr": "Welcome to your HR Mock Interview! I am your HR Interview Coach. I will ask you standard behavioral questions, evaluate your responses, suggest STAR improvements, and follow up. Let's begin!\n\n**First Question: Could you please tell me about yourself and what motivated you to study engineering?**",
        "practice_tech": "Welcome to the Technical Interview Preparation portal. I act as a Senior Technical Lead. We'll practice algorithms, data structures, databases, and system design. Let's start with a core concept:\n\n**Explain the differences between a Process and a Thread in terms of memory sharing. When would you use which?**",
        "practice_behav": "Welcome to the Behavioral Interview room. I will test your collaboration, conflict resolution, and leadership skills. Let's begin:\n\n**Describe a situation where you worked on a team project and had a major disagreement with a teammate. How did you resolve it?**",
        "practice_company": "Welcome to the target placement mock session. Please tell me which company you are preparing for (e.g. Amazon, Google, Microsoft) and we will initiate customized questions based on their selection criteria!",
        "learning": "Hello! I am your personal tutor. Ask me to explain any computer science topic (like Dynamic Programming, ML models, Joins, or OOPs pillars) or share a coding doubt, and I will teach it to you step-by-step with markdown code formatting! What shall we learn first?"
    }
    
    greeting = greetings.get(mode, greetings["chat"])
    
    # Save greeting in DB
    bot_msg = ChatMessage(
        session_id=new_sess.id,
        sender="ai",
        content=greeting
    )
    db.session.add(bot_msg)
    db.session.commit()
    
    flash("New HireWise Mentor session created successfully.", "success")
    return redirect(url_for('mentor.chat', session_id=new_sess.id))

@mentor_bp.route('/mentor/session/delete/<int:session_id>', methods=['POST'])
@login_required
def delete_session(session_id):
    """Deletes an entire chat session and its nested records."""
    sess = db.first_or_404(db.session.query(ChatSession).filter_by(id=session_id, user_id=current_user.id))
    db.session.delete(sess)
    db.session.commit()
    flash("Session deleted successfully.", "success")
    return redirect(url_for('mentor.chat'))

@mentor_bp.route('/mentor/session/<int:session_id>/messages', methods=['GET'])
@login_required
def get_messages(session_id):
    """API endpoint to load messages for a specific session via AJAX."""
    sess = db.first_or_404(db.session.query(ChatSession).filter_by(id=session_id, user_id=current_user.id))
    messages = db.session.query(ChatMessage).filter_by(session_id=sess.id).order_by(ChatMessage.created_at.asc()).all()
    
    result = []
    for m in messages:
        result.append({
            "id": m.id,
            "sender": m.sender,
            "content": m.content,
            "is_audio": m.is_audio,
            "audio_path": m.audio_path,
            "created_at": m.created_at.strftime('%I:%M %p')
        })
        
    return jsonify({"success": True, "messages": result, "mode": sess.mode})

@mentor_bp.route('/mentor/message/send', methods=['POST'])
@login_required
def send_message():
    """Ingests a text message, sends the history context to the LLM, saves the AI reply, and updates practice metrics."""
    session_id = request.json.get('session_id')
    content = request.json.get('content', '').strip()
    
    if not session_id or not content:
        return jsonify({"success": False, "error": "Invalid payload."}), 400
        
    sess = db.first_or_404(db.session.query(ChatSession).filter_by(id=session_id, user_id=current_user.id))
    
    # Save user message
    user_msg = ChatMessage(
        session_id=sess.id,
        sender="user",
        content=content
    )
    db.session.add(user_msg)
    db.session.commit()
    
    # Increment practice time by 30 seconds per chat message
    current_user.total_practice_time = (current_user.total_practice_time or 0) + 30
    db.session.commit()
    
    # Fetch conversational history for context
    db_messages = db.session.query(ChatMessage).filter_by(session_id=sess.id).order_by(ChatMessage.created_at.asc()).all()
    history = [{"sender": m.sender, "content": m.content} for m in db_messages]
    
    # Ingest resume if uploaded for personalized context
    resume_context = None
    user_resume_pattern = os.path.join(Config.UPLOAD_FOLDER, f"user_{current_user.id}_resume_*.pdf")
    resume_files = glob.glob(user_resume_pattern)
    if resume_files:
        try:
            resume_text = resume_service.extract_text(resume_files[0])
            if resume_text and len(resume_text.strip()) > 50:
                # Provide brief extracted details
                resume_context = f"Candidate Resume Tailoring details (Reference projects/skills if relevant):\n{resume_text[:2000]}"
        except Exception as e:
            print(f"Error parsing resume for conversational context: {e}")
            
    # If mode is company-specific, pass targeting information
    system_context = resume_context
    if sess.mode == "practice_company" and sess.title != "New Mentor Chat":
        system_context = f"Target Company: {sess.title}. " + (resume_context or "")
        
    # Generate LLM response
    import time
    start_time = time.time()
    ai_reply, provider, suggestions = mentor_service.generate_response(sess.mode, history, system_context)
    duration = time.time() - start_time
    
    provider_name = "Fallback"
    if provider == "groq":
        provider_name = "Groq"
    elif provider == "claude":
        provider_name = "Claude"
        
    print(f"\n[CHAT]\nProvider: {provider_name}\nPrompt: {content}\nResponse: {ai_reply}\nTime: {duration:.1f} sec\n")
    
    # Save AI response
    ai_msg = ChatMessage(
        session_id=sess.id,
        sender="ai",
        content=ai_reply
    )
    db.session.add(ai_msg)
    db.session.commit()
    
    # Update session message count & duration estimation (30 seconds per message)
    msg_count = db.session.query(ChatMessage).filter_by(session_id=sess.id).count()
    sess.num_messages = msg_count
    sess.duration = msg_count * 30
    db.session.commit()
    
    # Process practice score & records if mock mode
    if sess.mode.startswith("practice") and len(db_messages) >= 2:
        try:
            # Previous AI message is the question, User current is the answer
            prev_ai_msg = db_messages[-2]
            
            practice_score = 75.0
            # Try to extract scores like 85/100 or 8.5/10
            score_match = re.search(r'score:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)', ai_reply.lower())
            if score_match:
                val = float(score_match.group(1))
                total = float(score_match.group(2))
                practice_score = round((val / total) * 100, 1)
            else:
                pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', ai_reply)
                if pct_match:
                    practice_score = float(pct_match.group(1))
                    
            practice_rec = PracticeHistory(
                user_id=current_user.id,
                session_id=sess.id,
                topic=sess.mode.replace("practice_", "").upper(),
                question=prev_ai_msg.content,
                answer=content,
                feedback=ai_reply,
                score=practice_score
            )
            db.session.add(practice_rec)
            db.session.commit()
        except Exception as ex:
            print(f"Error logging practice history metrics: {ex}")
            db.session.rollback()
            
    return jsonify({
        "success": True,
        "reply": ai_reply,
        "provider": provider,
        "suggestions": suggestions,
        "time": ai_msg.created_at.strftime('%I:%M %p')
    })

@mentor_bp.route('/mentor/voice_transcribe', methods=['POST'])
@login_required
def voice_transcribe():
    """Ingests an uploaded audio blob, runs speech-to-text, and returns the transcript text."""
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file provided."}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"success": False, "error": "Empty audio filename."}), 400
        
    # Create temp directory
    temp_dir = Config.UPLOAD_FOLDER / "temp_voice"
    temp_dir.mkdir(exist_ok=True)
    
    filename = f"user_{current_user.id}_voice_{uuid.uuid4().hex[:8]}.webm"
    dest_path = temp_dir / filename
    
    try:
        audio_file.save(str(dest_path))
        
        # Transcribe using Whisper
        transcript_text = whisper_service.transcribe(dest_path)
        
        # Clean up temp file
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        return jsonify({
            "success": True,
            "transcript": transcript_text
        })
    except Exception as e:
        print(f"Error transcribing voice chat message: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@mentor_bp.route('/api/chat/provider', methods=['GET'])
@login_required
def get_chat_provider():
    """Returns the currently active conversational AI provider based on initialization status."""
    if mentor_service.groq_initialized:
        return jsonify({
            "provider": "groq",
            "status": "active",
            "label": "Groq Llama 3 (70B)"
        })
    elif mentor_service.claude_key:
        return jsonify({
            "provider": "claude",
            "status": "active",
            "label": "Claude"
        })
    else:
        return jsonify({
            "provider": "fallback",
            "status": "active",
            "label": "Local Practice Mode"
        })

@mentor_bp.route('/mentor/message/regenerate', methods=['POST'])
@login_required
def regenerate_message():
    """Removes the last AI response from the session and triggers a new LLM generation based on the user's last query."""
    session_id = request.json.get('session_id')
    
    if not session_id:
        return jsonify({"success": False, "error": "Invalid session ID."}), 400
        
    sess = db.first_or_404(db.session.query(ChatSession).filter_by(id=session_id, user_id=current_user.id))
    
    # Fetch messages in order
    db_messages = db.session.query(ChatMessage).filter_by(session_id=sess.id).order_by(ChatMessage.created_at.asc()).all()
    
    if not db_messages or db_messages[-1].sender != "ai":
        return jsonify({"success": False, "error": "No AI response found to regenerate."}), 400
        
    # Delete the last AI message
    last_ai_msg = db_messages[-1]
    db.session.delete(last_ai_msg)
    db.session.commit()
    
    # Reload messages for context (the last message is now the user's query)
    db_messages = db.session.query(ChatMessage).filter_by(session_id=sess.id).order_by(ChatMessage.created_at.asc()).all()
    history = [{"sender": m.sender, "content": m.content} for m in db_messages]
    
    # Ingest resume if uploaded
    resume_context = None
    user_resume_pattern = os.path.join(Config.UPLOAD_FOLDER, f"user_{current_user.id}_resume_*.pdf")
    resume_files = glob.glob(user_resume_pattern)
    if resume_files:
        try:
            resume_text = resume_service.extract_text(resume_files[0])
            if resume_text and len(resume_text.strip()) > 50:
                resume_context = f"Candidate Resume Tailoring details (Reference projects/skills if relevant):\n{resume_text[:2000]}"
        except Exception as e:
            print(f"Error parsing resume: {e}")
            
    system_context = resume_context
    if sess.mode == "practice_company" and sess.title != "New Mentor Chat":
        system_context = f"Target Company: {sess.title}. " + (resume_context or "")
        
    # Generate new response
    ai_reply, provider, suggestions = mentor_service.generate_response(sess.mode, history, system_context)
    
    # Save new AI response
    new_ai_msg = ChatMessage(
        session_id=sess.id,
        sender="ai",
        content=ai_reply
    )
    db.session.add(new_ai_msg)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "reply": ai_reply,
        "provider": provider,
        "suggestions": suggestions,
        "time": new_ai_msg.created_at.strftime('%I:%M %p')
    })

@mentor_bp.route('/mentor/session/<int:session_id>/rename', methods=['POST'])
@login_required
def rename_session(session_id):
    """Renames an existing chat session title."""
    sess = db.first_or_404(db.session.query(ChatSession).filter_by(id=session_id, user_id=current_user.id))
    new_title = request.json.get('title', '').strip()
    
    if not new_title:
        return jsonify({"success": False, "error": "Title cannot be empty."}), 400
        
    if len(new_title) > 80:
        new_title = new_title[:80] + "..."
        
    sess.title = new_title
    db.session.commit()
    
    return jsonify({"success": True, "title": sess.title})

@mentor_bp.route('/mentor/debug', methods=['GET'])
@login_required
def mentor_debug():
    """Renders developer debug panel for LLM telemetry and fallbacks."""
    data = LAST_DEBUG_INFO.get(current_user.id, {
        "current_provider": "local",
        "model": "rule-based-agent",
        "prompt_sent": "No prompts sent yet in this session.",
        "history_count": 0,
        "generated_response": "N/A",
        "fallback_reason": "N/A",
        "errors": "None"
    })
    
    if request.args.get('format') == 'json':
        return jsonify(data)
        
    return render_template('mentor_debug.html', debug=data)
