import os
import json
import re
import time
from config import Config

# Try to import groq, handle import error gracefully
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("WARNING: groq is not installed. Using rule-based fallback evaluations.")

class GroqService:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.model_name = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        
        # 1. Print Key Detection Logs
        if self.api_key:
            print("[INFO] GROQ_API_KEY detected in GroqService.")
        else:
            print("[WARNING] GROQ_API_KEY missing in GroqService.")
            
        # 2. Verify Groq integration
        self.initialized = False
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key, max_retries=2)
                self.initialized = True
                print("[INFO] GroqService initialized successfully.")
            except Exception as e:
                print(f"[ERROR] GroqService initialization failed:\n{e}")
                self.initialized = False

    def _create_chat_completion(self, messages, json_mode=False, temperature=0.2, max_tokens=None):
        """Calls Groq API with automatic fallback across available models, task-tuned temperature, and token controls."""
        models_to_try = [
            self.model_name,
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]
        unique_models = list(dict.fromkeys(models_to_try))
        
        last_error = None
        for m in unique_models:
            kwargs = {
                "model": m,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "invalid_api_key" in err_str:
                    print("[ERROR] Groq Authentication Failed: GROQ_API_KEY in .env is invalid or expired.")
                    raise e
                last_error = e
                print(f"[INFO] Groq model '{m}' failed ({e}). Trying next model...")
                continue
                
        raise last_error

    def _log_api_call(self, provider, endpoint, response_time, is_success, status_code=200, response_payload=None):
        """Helper to write diagnostic API logs to database"""
        try:
            from flask import current_app, has_app_context
            if not has_app_context():
                return
            from database.connection import db
            from models.api_log import APILog
            
            with current_app.app_context():
                log = APILog(
                    provider=provider,
                    endpoint=endpoint,
                    response_time=response_time,
                    is_success=is_success,
                    status_code=status_code,
                    response_payload=response_payload[:2000] if response_payload else None
                )
                db.session.add(log)
                db.session.commit()
        except Exception as ex:
            print(f"Failed to write API Log in GroqService: {ex}")

    def evaluate_answer(self, question, answer, category, expected_keywords=None):
        """
        Evaluates an answer based on Relevance, Clarity, Completeness, Structure, Professionalism.
        Provides strict, objective engineering assessment with no score inflation.
        """
        if not answer or len(answer.strip()) < 5:
            return self._generate_empty_evaluation()
            
        if self.initialized:
            start_time = time.time()
            try:
                keywords_context = f"Expected Core Technical Concepts: {expected_keywords}" if expected_keywords else "Expected: Accurate domain terminology and mechanisms."
                prompt = f"""
You are an expert technical interviewer and senior engineering hiring manager conducting an objective, rigorous assessment of a candidate's answer.

Interview Context:
- Category: {category}
- Question: "{question}"
- Candidate's Actual Answer: "{answer}"
- {keywords_context}

Evaluate the candidate's response with strict engineering objectivity across these 5 criteria:
1. relevance (0.0 to 10.0): Did the candidate answer the specific question asked? Penalize tangential filler or memorized unrelated scripts.
2. clarity (0.0 to 10.0): Is the explanation crisp, logically articulated, and free from confusing ambiguity?
3. completeness (0.0 to 10.0): Are essential technical components, internal mechanisms, or trade-offs covered?
4. structure (0.0 to 10.0): Does the response follow a coherent structure (core concept definition -> underlying mechanics -> practical trade-offs/applications)?
5. professionalism (0.0 to 10.0): Does the candidate use industry-standard engineering terminology and a professional tone?

Strict Scoring Calibration:
- 9.0–10.0: Staff/Senior level. Flawless, precise, deep mechanism knowledge and trade-offs.
- 7.5–8.9: Strong, technically sound, minor omissions in secondary edge cases.
- 5.0–7.4: Fair/Average. Knows basic definition but lacks underlying depth or relies on vague generalities.
- 3.0–4.9: Weak. Buzzword-heavy, contains notable technical inaccuracies or major omissions.
- 1.0–2.9: Poor. Factually wrong, completely evasive, or empty.

Important Rules:
- Never fabricate statements. Evaluate strictly what the candidate stated.
- Do not inflate scores. Concise answers that are technically complete should score high. Lengthy answers with fluff should NOT score high.
- The "feedback" must explicitly state: (a) what was answered correctly, (b) specific technical gaps or inaccuracies, and (c) an example of how a senior engineer would succinctly answer this.
- The "suggestions" must provide 2-3 specific, actionable technical takeaways.

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "relevance": 0.0,
  "clarity": 0.0,
  "completeness": 0.0,
  "structure": 0.0,
  "professionalism": 0.0,
  "feedback": "Detailed, constructive feedback outlining strengths, inaccuracies, and senior-level insights...",
  "suggestions": "1. Specific action point 1...\\n2. Specific action point 2..."
}}
"""
                
                response = self._create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=False,
                    temperature=0.15,
                    max_tokens=900
                )
                duration = time.time() - start_time
                response_text = response.choices[0].message.content.strip()
                
                # Sanitize response if it contains markdown code blocks
                if response_text.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
                    if match:
                        response_text = match.group(1).strip()
                
                eval_data = json.loads(response_text)
                required_keys = ["relevance", "clarity", "completeness", "structure", "professionalism", "feedback", "suggestions"]
                if all(k in eval_data for k in required_keys):
                    # Sanitize suggestions to string if returned as list
                    if isinstance(eval_data["suggestions"], list):
                        eval_data["suggestions"] = "\n".join(f"{i+1}. {s}" for i, s in enumerate(eval_data["suggestions"]))
                    for k in ["relevance", "clarity", "completeness", "structure", "professionalism"]:
                        eval_data[k] = round(float(eval_data[k]), 1)
                    self._log_api_call(provider="groq", endpoint="evaluate_answer", response_time=duration, is_success=True)
                    return eval_data
            except Exception as e:
                duration = time.time() - start_time
                self._log_api_call(provider="groq", endpoint="evaluate_answer", response_time=duration, is_success=False, status_code=500, response_payload=str(e))
                print(f"Groq API evaluation failed: {e}. Falling back to rule-based evaluation.")
                
        # Run Rule-based fallback if API is not available/failed
        return self._rule_based_evaluate(question, answer, category, expected_keywords)

    def evaluate_answer_adaptive(self, question, answer, category, skill=None, expected_keywords=None):
        """
        Evaluates an answer for adaptive technical interviews with research-grade objectivity.
        Grades 6 research-grade metrics without artificial inflation.
        """
        if not answer or len(answer.strip()) < 5:
            return {
                "correctness": 1.0,
                "depth": 1.0,
                "relevance": 1.0,
                "completeness": 1.0,
                "communication_quality": 1.0,
                "confidence": 1.0,
                "answer_score": 10.0,
                "feedback": "No substantial response was provided to evaluate.",
                "suggestions": "Please answer the question by speaking clearly into the microphone or typing your answer."
            }
            
        if self.initialized:
            start_time = time.time()
            try:
                skill_str = f" for the skill '{skill}'" if skill else ""
                keywords_context = f"Target Domain Keywords: {expected_keywords}" if expected_keywords else ""
                prompt = f"""
You are an expert technical interviewer evaluating an adaptive technical interview answer with research-grade rigor.

Interview Parameters:
- Category: {category}{skill_str}
- Question: "{question}"
- Candidate's Actual Answer: "{answer}"
- {keywords_context}

Evaluate the response across the following 6 research dimensions (each from 0.0 to 10.0):
1. correctness (0.0 to 10.0): Is the technical explanation factually correct? Identify any misconceptions, invalid syntax, or false claims.
2. depth (0.0 to 10.0): Does the candidate explain "how" and "why" — internal architecture, memory/time complexity (O notation), algorithms, or data structures?
3. relevance (0.0 to 10.0): Does the response directly address the question without irrelevant rambling?
4. completeness (0.0 to 10.0): Are all critical components and common edge cases addressed?
5. communication_quality (0.0 to 10.0): Is the response well-articulated, cohesive, and structured logically?
6. confidence (0.0 to 10.0): Is the tone assertive and definitive, avoiding excessive hesitation or tentative qualifiers ("maybe", "I guess")?

Overall Score Calibration:
- Calculate "answer_score" (0.0 to 100.0) reflecting overall competency:
  * 90–100: Exceptional, deep technical grasp with trade-offs and complexity analysis.
  * 75–89: Solid, correct, covers primary mechanics with minor omissions.
  * 50–74: Average, partial understanding, relies on surface definition without mechanics.
  * 25–49: Deficient, significant misconceptions or missing core concept.
  * 0–24: Inaccurate, off-topic, or non-responsive.

Crucial Instructions:
- Base feedback entirely on the candidate's actual text. Do NOT invent concepts or praise what was not mentioned.
- "feedback": 1–2 paragraphs detailing exact strengths, specific errors/omissions, and a model reference answer.
- "suggestions": 2–3 actionable, bulleted technical improvements.

Respond strictly with a valid JSON object matching this exact schema:
{{
  "correctness": 0.0,
  "depth": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "communication_quality": 0.0,
  "confidence": 0.0,
  "answer_score": 0.0,
  "feedback": "Objective, detailed feedback detailing exact strengths, gaps, and technical insights...",
  "suggestions": "1. Concrete improvement 1...\\n2. Concrete improvement 2..."
}}
"""
                
                response = self._create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=False,
                    temperature=0.15,
                    max_tokens=900
                )
                duration = time.time() - start_time
                response_text = response.choices[0].message.content.strip()
                
                # Sanitize response if it contains markdown code blocks
                if response_text.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
                    if match:
                        response_text = match.group(1).strip()
                
                eval_data = json.loads(response_text)
                required_keys = ["correctness", "depth", "relevance", "completeness", "communication_quality", "confidence", "answer_score", "feedback", "suggestions"]
                if all(k in eval_data for k in required_keys):
                    if isinstance(eval_data["suggestions"], list):
                        eval_data["suggestions"] = "\n".join(f"{i+1}. {s}" for i, s in enumerate(eval_data["suggestions"]))
                    for k in ["correctness", "depth", "relevance", "completeness", "communication_quality", "confidence"]:
                        eval_data[k] = round(float(eval_data[k]), 1)
                    eval_data["answer_score"] = round(float(eval_data["answer_score"]), 1)
                    self._log_api_call(provider="groq", endpoint="evaluate_answer_adaptive", response_time=duration, is_success=True)
                    return eval_data
            except Exception as e:
                duration = time.time() - start_time
                self._log_api_call(provider="groq", endpoint="evaluate_answer_adaptive", response_time=duration, is_success=False, status_code=500, response_payload=str(e))
                print(f"Groq API adaptive evaluation failed: {e}. Falling back to rule-based adaptive evaluation.")
                
        # Run Rule-based fallback if API is not available/failed
        return self._rule_based_evaluate_adaptive(question, answer, category, skill, expected_keywords)

    def _rule_based_evaluate_adaptive(self, question, answer, category, skill=None, expected_keywords=None):
        """
        Rule-based technical evaluator that scores answers when Groq API is unavailable.
        """
        words = answer.split()
        word_count = len(words)
        
        # Helper: load expected keywords
        keywords = []
        if expected_keywords:
            if isinstance(expected_keywords, str):
                try:
                    keywords = json.loads(expected_keywords)
                except Exception:
                    keywords = []
            elif isinstance(expected_keywords, list):
                keywords = expected_keywords
                
        # Heuristics:
        matched_keywords = []
        relevance_score = 5.0
        if keywords:
            for kw in keywords:
                if kw.lower() in answer.lower():
                    matched_keywords.append(kw)
            keyword_ratio = len(matched_keywords) / len(keywords) if len(keywords) > 0 else 1.0
            relevance_score += (keyword_ratio * 5.0)
        else:
            q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
            ans_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            overlap = q_words.intersection(ans_words)
            overlap_ratio = len(overlap) / len(q_words) if len(q_words) > 0 else 0.5
            relevance_score += min(5.0, overlap_ratio * 7.5)
            
        correctness_score = 4.0
        if matched_keywords:
            correctness_score += min(5.0, len(matched_keywords) * 1.5)
        else:
            correctness_score += min(4.0, (word_count / 15) * 1.0)
        correctness_score = min(10.0, correctness_score)
        
        depth_score = 3.0
        depth_words = ["because", "how", "why", "mechanism", "underlying", "process", "components", "complexity", "internal", "structure", "explain"]
        depth_matches = [w for w in depth_words if w in answer.lower()]
        depth_score += min(4.0, len(depth_matches) * 1.0)
        if word_count > 50:
            depth_score += 1.5
        if word_count > 100:
            depth_score += 1.5
        depth_score = min(10.0, depth_score)
        
        completeness_score = 3.0
        if word_count > 25:
            completeness_score += 2.0
        if word_count > 60:
            completeness_score += 2.0
        if word_count > 110:
            completeness_score += 3.0
        completeness_score = min(10.0, completeness_score)
        
        comm_score = 5.0
        transitions = ["firstly", "secondly", "finally", "therefore", "however", "consequently", "for instance", "to summarize"]
        trans_matches = [t for t in transitions if t in answer.lower()]
        comm_score += min(3.0, len(trans_matches) * 1.0)
        if word_count > 30:
            comm_score += 2.0
        comm_score = min(10.0, comm_score)
        
        qualifiers = ["maybe", "i guess", "probably", "not sure", "don't know", "dont know", "perhaps", "i think"]
        qualifier_count = sum(1 for q in qualifiers if q in answer.lower())
        confidence_score = 9.0 - (qualifier_count * 1.5)
        confidence_score = max(2.0, min(10.0, confidence_score))
        
        if word_count < 10:
            relevance_score = max(1.0, relevance_score - 4.0)
            correctness_score = max(1.0, correctness_score - 4.0)
            depth_score = max(1.0, depth_score - 4.0)
            completeness_score = max(1.0, completeness_score - 4.0)
            comm_score = max(1.0, comm_score - 4.0)
            confidence_score = max(1.0, confidence_score - 4.0)
            
        relevance_score = round(relevance_score, 1)
        correctness_score = round(correctness_score, 1)
        depth_score = round(depth_score, 1)
        completeness_score = round(completeness_score, 1)
        comm_score = round(comm_score, 1)
        confidence_score = round(confidence_score, 1)
        
        avg_score = (relevance_score + correctness_score + depth_score + completeness_score + comm_score + confidence_score) / 6.0
        answer_score = round(avg_score * 10.0, 1)
        
        feedback = f"Rule-based Technical evaluation. Matched key concepts: {', '.join(matched_keywords) if matched_keywords else 'None'}. "
        if depth_score < 6.0:
            feedback += "The answer lacks technical depth. Try explaining 'how' things work internally. "
        else:
            feedback += "The answer demonstrates a good grasp of the technical concepts. "
        if confidence_score < 7.0:
            feedback += "Try to sound more confident and avoid tentative phrases."
            
        suggestions = "1. Detail the internal mechanics or time/space complexity.\n2. Structure using transition words like 'First', 'Then', 'Finally'.\n3. Avoid tentative expressions such as 'I think' or 'maybe'."
        
        return {
            "correctness": correctness_score,
            "depth": depth_score,
            "relevance": relevance_score,
            "completeness": completeness_score,
            "communication_quality": comm_score,
            "confidence": confidence_score,
            "answer_score": answer_score,
            "feedback": feedback,
            "suggestions": suggestions
        }

    def generate_follow_up(self, question, answer):
        """
        Generates a dynamic, highly-targeted follow-up question based directly on the candidate's actual answer.
        Probes missing mechanisms, challenges complete answers with scaling/edge cases, or explores trade-offs.
        """
        default_follow_up = "Could you elaborate on how you handled the architectural challenges or complexity trade-offs during that implementation?"
        
        if self.initialized:
            start_time = time.time()
            try:
                prompt = f"""
You are an experienced senior technical interviewer conducting an in-depth mock interview. The candidate has just answered a question.

Original Question: "{question}"
Candidate's Actual Answer: "{answer}"

Generate a single, natural, highly-targeted follow-up question based directly on their response:
- If their answer was incomplete or surface-level: Probe the specific missing mechanism, underlying data structure, or complexity trade-off ($O$ notation).
- If their answer was strong and complete: Challenge them with an advanced scenario (e.g. high-concurrency scaling, memory bottlenecks, failure recovery, or edge cases).
- If their answer contained a misconception: Ask a constructive, targeted question that prompts them to examine that specific principle without sounding argumentative.
- If they mentioned a specific tool, framework, or algorithm: Ask about their implementation experience or why they would choose it over a common alternative.

Requirements:
- Ask exactly ONE question.
- Keep it concise (1 to 2 sentences max).
- Sound like an engaging, curious engineering lead.
- Do NOT include conversational pleasantries, introductory praise, or filler ("Great answer!", "Thank you"). Output ONLY the follow-up question itself.
"""
                
                response = self._create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=False,
                    temperature=0.6,
                    max_tokens=150
                )
                duration = time.time() - start_time
                follow_up_text = response.choices[0].message.content.strip().strip('"\'')
                # Clean any accidental conversational prefix
                follow_up_text = re.sub(r'^(?:follow-up question:?|here is a follow-up:?)\s*', '', follow_up_text, flags=re.IGNORECASE)
                if follow_up_text:
                    self._log_api_call(provider="groq", endpoint="generate_follow_up", response_time=duration, is_success=True)
                    return follow_up_text
            except Exception as e:
                duration = time.time() - start_time
                self._log_api_call(provider="groq", endpoint="generate_follow_up", response_time=duration, is_success=False, status_code=500, response_payload=str(e))
                print(f"Groq API follow-up generation failed: {e}")
                
        # Fallback logic based on answer keywords
        answer_lower = answer.lower()
        if "project" in answer_lower or "built" in answer_lower:
            return "That sounds interesting. Could you tell me more about your specific role in that project and the technologies you chose?"
        elif "algorithm" in answer_lower or "code" in answer_lower or "complex" in answer_lower:
            return "How did you optimize your solution, and what was its time and space complexity?"
        elif "team" in answer_lower or "conflict" in answer_lower or "collaboration" in answer_lower:
            return "Looking back, is there anything you would have handled differently to achieve a faster or better resolution?"
        
        return default_follow_up

    def generate_resume_questions(self, resume_text):
        """
        Parses candidate resume text and generates 5 progressive, rigorous interview questions
        tailored directly to their declared stack, projects, and career domain.
        """
        fallback_questions = [
            "Explain the architectural decisions and component trade-offs of the primary project listed on your resume.",
            "Walk me through the internal mechanism of the core language/database you claimed highest proficiency with.",
            "Can you describe a challenging engineering bottleneck you encountered in production and how you debugged it?",
            "How would you re-architect your favorite project to support a 10x surge in concurrent traffic?",
            "Describe a situation where you had to balance technical debt with delivering a feature on time."
        ]
        
        if not resume_text or len(resume_text.strip()) < 50:
            return fallback_questions
            
        if self.initialized:
            start_time = time.time()
            try:
                prompt = f"""
You are an engineering hiring manager designing a technical interview tailored specifically to a candidate's resume.

Resume Content:
\"\"\"{resume_text[:4000]}\"\"\"

Analyze the resume's stated technologies, frameworks, and projects. Generate exactly 5 progressive, rigorous interview questions that evaluate genuine technical depth:
1. Architecture & Design: A deep dive into the architecture, component breakdown, and database decisions of their most significant stated project.
2. Core Technical Competency: An in-depth mechanism question on their primary claimed programming language or backend/frontend framework (e.g., memory management, asynchronous execution, indexing, state synchronization).
3. Engineering Challenge & Trade-offs: A question probing a real technical bottleneck, scalability challenge, or engineering trade-off faced during project development.
4. Production Readiness & Reliability: A scenario-based question about handling failures, caching, API security, or high concurrent load in their project domain.
5. Behavioral & Ownership (STAR format): A question about technical ownership, resolving a challenging bug, or code review collaboration tied to their stated experience.

Respond strictly with a valid JSON array of 5 strings. Do not add markdown code blocks or explanatory text.
Example format:
[
  "In your [Project Name], walk me through how data flows from the client to your database...",
  "Given your experience with [Technology], how does it handle..."
]
"""
                response = self._create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=False,
                    temperature=0.5,
                    max_tokens=700
                )
                duration = time.time() - start_time
                response_text = response.choices[0].message.content.strip()
                
                if response_text.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
                    if match:
                        response_text = match.group(1).strip()
                        
                questions = json.loads(response_text)
                if isinstance(questions, list) and len(questions) > 0:
                    clean_questions = [str(q).strip() for q in questions if str(q).strip()]
                    if len(clean_questions) >= 3:
                        self._log_api_call(provider="groq", endpoint="generate_resume_questions", response_time=duration, is_success=True)
                        return clean_questions[:5]
            except Exception as e:
                duration = time.time() - start_time
                self._log_api_call(provider="groq", endpoint="generate_resume_questions", response_time=duration, is_success=False, status_code=500, response_payload=str(e))
                print(f"Groq API resume questions generation failed: {e}")
                print(f"Groq API resume questions generation failed: {e}")
                
        # Simple local parser if API is not setup
        extracted_questions = []
        skills_matched = []
        tech_keywords = ["python", "javascript", "react", "sql", "java", "c++", "machine learning", "django", "flask", "node"]
        for tech in tech_keywords:
            if tech in resume_text.lower():
                skills_matched.append(tech.capitalize())
        
        if skills_matched:
            skills_str = ", ".join(skills_matched[:3])
            extracted_questions.append(f"I see you have skills in {skills_str}. Can you talk about a time when you combined these technologies to solve a problem?")
        
        project_matches = re.findall(r'(?:project|portfolio|built|developed)\b', resume_text, re.IGNORECASE)
        if len(project_matches) > 0:
            extracted_questions.append("Based on the projects on your resume, explain the main challenge you faced during development and how you overcame it.")
            extracted_questions.append("If you were to rebuild your primary project today, what design patterns or architectural decisions would you change?")
            
        for q in fallback_questions:
            if len(extracted_questions) < 5:
                extracted_questions.append(q)
                
        return extracted_questions[:5]

    def _generate_empty_evaluation(self):
        return {
            "relevance": 1.0,
            "clarity": 1.0,
            "completeness": 1.0,
            "structure": 1.0,
            "professionalism": 1.0,
            "feedback": "No substantial response was provided to evaluate.",
            "suggestions": "Please answer the question by speaking clearly into the microphone or typing your answer."
        }

    def _rule_based_evaluate(self, question, answer, category, expected_keywords=None):
        """
        Rule-based NLP evaluator that scores answers when Groq API is unavailable.
        """
        words = answer.split()
        word_count = len(words)
        
        keywords = []
        if expected_keywords:
            if isinstance(expected_keywords, str):
                try:
                    keywords = json.loads(expected_keywords)
                except Exception:
                    keywords = []
            elif isinstance(expected_keywords, list):
                keywords = expected_keywords
                
        matched_keywords = []
        relevance_score = 5.0
        if keywords:
            for kw in keywords:
                if kw.lower() in answer.lower():
                    matched_keywords.append(kw)
            keyword_ratio = len(matched_keywords) / len(keywords) if len(keywords) > 0 else 1.0
            relevance_score += (keyword_ratio * 5.0)
        else:
            q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
            ans_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            overlap = q_words.intersection(ans_words)
            overlap_ratio = len(overlap) / len(q_words) if len(q_words) > 0 else 0.5
            relevance_score += min(5.0, overlap_ratio * 7.5)
            
        clarity_score = 6.0
        if word_count > 15:
            clarity_score += 2.0
        if '.' in answer or '?' in answer:
            clarity_score += 1.0
        if ',' in answer:
            clarity_score += 1.0
        clarity_score = min(10.0, clarity_score)
        
        completeness_score = 4.0
        if word_count > 30:
            completeness_score += 2.0
        if word_count > 60:
            completeness_score += 2.0
        if word_count > 100:
            completeness_score += 2.0
        completeness_score = min(10.0, completeness_score)
        
        structure_score = 5.0
        structuring_words = ["firstly", "secondly", "finally", "because", "however", "therefore", "for example", "instance", "conclusion", "summarize", "then", "next", "after that"]
        struct_matches = [w for w in structuring_words if w in answer.lower()]
        structure_score += min(5.0, len(struct_matches) * 1.5)
        
        professionalism_score = 6.0
        professional_words = ["system", "implement", "optimize", "deliver", "achieve", "design", "manage", "collaborate", "solve", "critical", "analyze", "strategy", "efficient"]
        prof_matches = [w for w in professional_words if w in answer.lower()]
        professionalism_score += min(4.0, len(prof_matches) * 1.0)
        
        if word_count < 10:
            relevance_score = max(2.0, relevance_score - 3.0)
            clarity_score = max(2.0, clarity_score - 3.0)
            completeness_score = max(2.0, completeness_score - 4.0)
            structure_score = max(2.0, structure_score - 3.0)
            professionalism_score = max(2.0, professionalism_score - 3.0)
            
        relevance_score = round(relevance_score, 1)
        clarity_score = round(clarity_score, 1)
        completeness_score = round(completeness_score, 1)
        structure_score = round(structure_score, 1)
        professionalism_score = round(professionalism_score, 1)
        
        feedback = f"Your answer of {word_count} words was analyzed by the HireWise rule engine. "
        if len(matched_keywords) > 0:
            feedback += f"You successfully referenced key terms: {', '.join(matched_keywords)}. "
        else:
            feedback += "Try to include more technical terms relevant to the question. "
            
        if len(struct_matches) >= 2:
            feedback += "Excellent logical structure was detected using transitional signposts."
        else:
            feedback += "The structural flow can be improved by dividing the response into a clear introduction, core points, and a concluding sentence."
            
        suggestions = "1. Try using the STAR (Situation, Task, Action, Result) methodology to format your responses.\n2. Incorporate more industry-standard terminology.\n3. Speak for at least 45-60 seconds to provide a comprehensive response."
        
        return {
            "relevance": relevance_score,
            "clarity": clarity_score,
            "completeness": completeness_score,
            "structure": structure_score,
            "professionalism": professionalism_score,
            "feedback": feedback,
            "suggestions": suggestions
        }

    def analyze_resume_details(self, resume_text):
        """
        Analyzes a candidate's resume with professional ATS and technical recruiter rigor.
        Extracts verified skills, projects, experience, missing high-demand industry skills,
        and provides calibrated ATS scores with Google X-Y-Z formula recommendations.
        """
        fallback_details = {
            "skills": "Software Development",
            "projects": "No specific projects found in text",
            "experience": "No specific experience found in text",
            "certifications": "No certifications listed in text",
            "missing_skills": "System Design, Cloud Deployments, Microservices, CI/CD",
            "ats_score": 60,
            "suggestions": "1. Convert bullet points using Google's X-Y-Z formula: Accomplished [X] as measured by [Y] by doing [Z].\n2. Add quantifiable engineering metrics (e.g., latency reduction, throughput, user count).\n3. Include missing industry-standard tools like Docker, Git workflows, and automated testing."
        }
        if not resume_text or len(resume_text.strip()) < 50:
            return fallback_details
            
        if self.initialized:
            start_time = time.time()
            try:
                prompt = f"""
You are a senior technical recruiter and ATS (Applicant Tracking System) optimization specialist. Parse and critically evaluate the candidate's resume text.

Resume Content:
\"\"\"{resume_text[:4000]}\"\"\"

Perform a rigorous, honest evaluation and return a valid JSON object matching this schema:
{{
  "skills": "Clean, comma-separated list of verified technical and professional skills detected in the resume",
  "projects": "Concise summary of candidate's technical projects including detected tech stack and features",
  "experience": "Summary of work experience, internships, or academic project milestones",
  "certifications": "Certifications, courses, or relevant academic honors detected (or 'None explicitly listed')",
  "missing_skills": "Top 3–5 high-demand industry skills missing for their specific career track (e.g., Docker, CI/CD, Redis, Unit Testing, System Design)",
  "ats_score": <integer between 35 and 90 calibrated as: 75–90 for strong quantifiable impact metrics using the Google X-Y-Z formula, clean formatting, and high keyword match; 55–74 for standard descriptive bullets without clear metrics; 35–54 for generic or sparse content>,
  "suggestions": "3–4 actionable, bulleted recommendations to optimize this resume for ATS screening. Specifically demonstrate how to convert at least one project bullet into Google's X-Y-Z formula (Accomplished [X] as measured by [Y] by doing [Z]). Note that scores are diagnostic benchmarks."
}}

Do not wrap in markdown code blocks. Respond ONLY with the JSON object.
"""
                response = self._create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=False,
                    temperature=0.2,
                    max_tokens=900
                )
                duration = time.time() - start_time
                response_text = response.choices[0].message.content.strip()
                
                if response_text.startswith("```"):
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
                    if match:
                        response_text = match.group(1).strip()
                        
                details = json.loads(response_text)
                required_keys = ["skills", "projects", "experience", "certifications", "missing_skills", "ats_score", "suggestions"]
                if all(k in details for k in required_keys):
                    try:
                        score_val = int(float(details["ats_score"]))
                        details["ats_score"] = max(20, min(95, score_val))
                    except Exception:
                        details["ats_score"] = 65
                    if isinstance(details["suggestions"], list):
                        details["suggestions"] = "\n".join(f"{i+1}. {s}" for i, s in enumerate(details["suggestions"]))
                    self._log_api_call(provider="groq", endpoint="analyze_resume_details", response_time=duration, is_success=True)
                    return details
            except Exception as e:
                duration = time.time() - start_time
                self._log_api_call(provider="groq", endpoint="analyze_resume_details", response_time=duration, is_success=False, status_code=500, response_payload=str(e))
                print(f"Groq API resume detailed analysis failed: {e}. Using rule-based fallback.")
                
        # Rule-based fallback parsing
        skills_found = []
        known_skills = ['python', 'javascript', 'html', 'css', 'react', 'node', 'sql', 'sqlite', 'mongodb', 'c++', 'java', 'opencv', 'mediapipe', 'whisper', 'flask', 'django', 'machine learning', 'scikit-learn']
        text_lower = resume_text.lower()
        for skill in known_skills:
            if skill in text_lower:
                skills_found.append(skill.title())
        skills_str = ", ".join(skills_found) if skills_found else "Software Development"
        
        projects = []
        lines = resume_text.split('\n')
        for line in lines:
            if any(p in line.lower() for p in ['project', 'portfolio', 'built', 'developed']):
                if len(line.strip()) > 15:
                    projects.append(line.strip())
        projects_str = "; ".join(projects[:3]) if projects else "General projects found in text"
        
        experience = []
        for line in lines:
            if any(e in line.lower() for e in ['experience', 'intern', 'work', 'job', 'employed', 'position']):
                if len(line.strip()) > 15:
                    experience.append(line.strip())
        experience_str = "; ".join(experience[:3]) if experience else "General experience found in text"

        return {
            "skills": skills_str,
            "projects": projects_str,
            "experience": experience_str,
            "certifications": "Certifications listed in text",
            "missing_skills": "System Design, Cloud Deployments, Microservices",
            "ats_score": 75 if len(skills_found) > 4 else 55,
            "suggestions": "1. Highlight technical achievements with measurable impact.\n2. Add standard industry keywords related to target roles.\n3. Keep formatting clean and consistent."
        }
