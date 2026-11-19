import os
import sys
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GROQ_API_KEY", "").strip().strip("\"'")
print("=" * 60)
print("GROQ API DIAGNOSTIC TEST")
print("=" * 60)
print(f"API Key found in .env: {bool(key)}")
if key:
    masked = key[:7] + "..." + key[-4:] if len(key) > 12 else "INVALID_LENGTH"
    print(f"Key preview: {masked} (length: {len(key)})")
else:
    print("[ERROR] GROQ_API_KEY is empty in .env!")
    sys.exit(1)

try:
    from groq import Groq
    client = Groq(api_key=key, max_retries=2)
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    print(f"\nSending test prompt to Groq API (model: {model})...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Respond with only the word: WORKING"}]
    )
    answer = resp.choices[0].message.content.strip()
    print(f"\n[SUCCESS] Groq API responded: {answer}")
    print("=" * 60)
    print("Groq is FULLY OPERATIONAL! Live AI calls will now work in HireWise AI.")
    print("=" * 60)
except Exception as e:
    err_str = str(e)
    print(f"\n[FAILURE] Groq API Call Failed:")
    if "401" in err_str or "invalid_api_key" in err_str:
        print(">> Reason: HTTP 401 - INVALID API KEY")
        print(">> The key in .env has expired or been revoked/deleted on Groq.")
        print(">> FIX: Go to https://console.groq.com/keys , create a free new key, and paste it into .env")
    elif "404" in err_str or "model_not_found" in err_str:
        print(f">> Reason: HTTP 404 - Model '{model}' not found or decommissioned.")
        print(">> FIX: Set GROQ_MODEL=qwen/qwen3.8-27b in .env")
    elif "429" in err_str:
        print(">> Reason: HTTP 429 - RATE LIMIT HIT")
    else:
        print(f">> Details: {e}")
    print("=" * 60)
