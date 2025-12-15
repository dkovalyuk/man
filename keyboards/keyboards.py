from aiogram import Bot
from aiogram.types import BotCommand
from lexicon.lexicon import LEXICON

async def setup_main_menu(bot: Bot):
    main_menu_commands = []
    for command, command_description in LEXICON.items():
        main_menu_commands.append(
            #BotCommand(command = "start", description = "Стартує бота"),
            BotCommand(command = command, description = command_description),
        )

    await bot.set_my_commands(main_menu_commands)