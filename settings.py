# # Take environment variables from .env.local file if it exists
import os


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

EETC_API_KEY = os.getenv("EETC_API_KEY")

AWS_EMAIL_SENDER = "eastempiretradingcompany2019@gmail.com"
AWS_REGION = "eu-central-1"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

try:
    from local_settings import *
except ImportError:
    pass
