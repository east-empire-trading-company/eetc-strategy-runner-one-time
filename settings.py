# # Take environment variables from .env.local file if it exists
import os


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

EETC_API_KEY = os.getenv("EETC_API_KEY")

try:
    from local_settings import *
except ImportError:
    pass
