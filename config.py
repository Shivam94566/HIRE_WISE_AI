import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if it exists
load_dotenv(BASE_DIR / ".env")

class Config:
    BASE_DIR = BASE_DIR
    # Flask Settings
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "hirewise_secure_fallback_secret_key_2026")
    ENV = os.environ.get("FLASK_ENV", "development")
    DEBUG = ENV == "development"

    # Database Settings
    SQLALCHEMY_DATABASE_URI = "sqlite:///database/hirwise.db"
    # Ensure parent directory of DB exists
    (BASE_DIR / "database").mkdir(parents=True, exist_ok=True)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads Configuration
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    ALLOWED_EXTENSIONS = {"pdf", "wav", "webm", "mp4", "mp3"}
    # Max file size: 30MB
    MAX_CONTENT_LENGTH = 30 * 1024 * 1024

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or ""
    CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    
    # Whisper settings
    WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "tiny")

    # Mail Server Configuration
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", "587"))
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "True").lower() in ("true", "1", "yes")
    ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@hirewise.ai")
