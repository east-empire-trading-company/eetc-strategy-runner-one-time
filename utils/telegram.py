# TODO extract Telegram message sending to separate microservice
import telegram as tg

import settings


class EETCTelegramChannel:
    def __init__(self):
        self._bot = tg.Bot(token=settings.TELEGRAM_BOT_TOKEN)

    async def send_message(self, message):
        await self._bot.send_message(
            chat_id=settings.TELEGRAM_CHANNEL_ID,
            text=message,
            parse_mode=tg.constants.ParseMode.HTML,
        )

    async def send_image(self, image, caption):
        await self._bot.send_photo(
            chat_id=settings.TELEGRAM_CHANNEL_ID,
            photo=image,
            caption=caption,
            write_timeout=120,
        )
