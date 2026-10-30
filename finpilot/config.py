"""Central configuration, read from environment / .env (never hardcoded secrets)."""
import os

from dotenv import load_dotenv

load_dotenv()

MESH_API_KEY = os.getenv("MESH_API_KEY", "")
MESH_BASE_URL = os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1")
MODEL_FAST = os.getenv("MESH_MODEL_FAST", "openai/gpt-4.1-mini")
MODEL_SMART = os.getenv("MESH_MODEL_SMART", "openai/gpt-4.1")

DB_PATH = os.getenv("FINPILOT_DB", "finpilot.db")
PROFILE_DIR = os.getenv("FINPILOT_PROFILE_DIR", "profiles")  # one SQLite file per profile

DIGEST_DAY_OF_WEEK = os.getenv("DIGEST_CRON_DAY_OF_WEEK", "mon")
DIGEST_HOUR = int(os.getenv("DIGEST_CRON_HOUR", "9"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

CATEGORIES = [
    "income", "rent", "emi", "groceries", "dining", "transport", "utilities",
    "subscriptions", "shopping", "entertainment", "health", "education", "travel",
    "transfers", "other",
]
# Spending the user can realistically cut; used by analysis + advice.
DISCRETIONARY = {"dining", "shopping", "entertainment", "subscriptions", "travel"}
