import telegram as tg

import settings


class EETCTelegramChannel:
    def __init__(self):
        self._bot = tg.Bot(token=settings.TELEGRAM_BOT_TOKEN)

    async def broadcast_message(self, message):
        await self._bot.send_message(
            chat_id=settings.TELEGRAM_CHANNEL_ID,
            text=message,
            parse_mode=tg.constants.ParseMode.HTML,
        )
