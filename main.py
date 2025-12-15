import asyncio
import logging
from aiogram import Bot, Dispatcher
from config.config import Config, load_config
from handlers.user import user_router
from keyboards.keyboards import setup_main_menu

# Ініціалізуємо логгер
logger = logging.getLogger(__name__)

# Функція конфігурації та запуску бота
async def main():
    # Завантажуємо конфіг в змінну config
    config = load_config()

    # Задаємо базову конфігурацію логування
    logging.basicConfig(
        level=logging.getLevelName(level=config.log.level),
        format=config.log.format,
    )   

    # Виводимо в консоль інформацію про початок запуску бота
    logger.info("Starting bot")

    # Ініціалізуємо бот і диспетчер
    bot = Bot(token=config.token)
    dp = Dispatcher()

    # Реєструємо роутери в диспетчері
    dp.include_router(user_router)

    # Пропускаємо накопичені оновлення і запускаємо polling, підключаємо боту меню
    await bot.delete_webhook(drop_pending_updates=True)
    await setup_main_menu(bot)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())