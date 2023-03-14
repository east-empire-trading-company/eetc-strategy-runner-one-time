# # Take environment variables from .env.local file if it exists
import os


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

EETC_API_KEY = os.getenv("EETC_API_KEY")
EETC_VAULT_API_KEY = os.getenv("EETC_VAULT_API_KEY")

AWS_EMAIL_SENDER = "eastempiretradingcompany2019@gmail.com"
AWS_REGION = "eu-central-1"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

IBKR_TWS_PORT = 7496
IB_GATEWAY_PORT = 4001
IBKR_TWS_HOST = "127.0.0.1"
IB_GATEWAY_HOST = "127.0.0.1"

try:
    from local_settings import *
except ImportError:
    pass
