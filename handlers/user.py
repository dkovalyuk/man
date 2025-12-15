import torch
import torch.nn as nn
import joblib
from aiogram import Router
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup, default_state


class Model(nn.Module):
    def __init__(self, input_factors):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_factors, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


class FMSPredict(StatesGroup):
    get_user_data = State()


user_router = Router()

# Завантажуємо scaler та модель
scaler = joblib.load("scaler.pkl")
model = Model(input_factors=11)

#використовуємо CPU, бо на хостингу немає CUDA 
model.load_state_dict(torch.load("cardio_quickstart_mlp.pth", map_location=torch.device("cpu")))
model.eval()


@user_router.message(CommandStart(), StateFilter(default_state))
async def start_command(message: Message):
    await message.answer(text="Привіт, я бот, який прогнозує артеріальну гіпертензію на базі нейронної мережі. Щоб розпочати, введіть команду /check")


@user_router.message(Command(commands="help"), StateFilter(default_state))
async def help_command(message: Message):
    await message.answer(text="Цей бот прогнозує артеріальну гіпертензію на основі нейронної мережі. Щоб розпочати прогнозування, скористайтеся командою /check. Якщо ви хочете вийти з режиму прогнозування, введіть команду /cancel")


@user_router.message(Command(commands="check"), StateFilter(default_state))
async def check_command(message: Message, state: FSMContext):
    text = (
        "Ви ввійшли в режим прогнозування артеріальної гіпертензії. Введіть 11 чисел через пробіл у такому порядку:\n\n"
        "1. Вік (роки)\n"
        "2. Стать (1 — жінка, 2 — чоловік)\n"
        "3. Зріст (см)\n"
        "4. Вага (кг)\n"
        "5. Верхній тиск (ap_hi)\n"
        "6. Нижній тиск (ap_lo)\n"
        "7. Холестерин (1–3) 1: нормальний, 2: вище норми, 3: значно вище норми\n"
        "8. Глюкоза (1–3) 1: нормальний, 2: вище норми, 3: значно вище норми\n"
        "9. Куріння (0/1)\n"
        "10. Вживання алкоголю (0/1)\n"
        "11. Фізична активність (0/1)\n\n"
        "Приклад: 55 1 170 75 120 80 2 1 0 0 1"
    )
    await message.answer(text=text)
    await state.set_state(FMSPredict.get_user_data)


@user_router.message(Command(commands="cancel"), ~StateFilter(default_state))
async def cancel_command_state(message: Message, state: FSMContext):
    await message.answer("Ви вийшли з режиму перевірки.")
    await state.clear()


@user_router.message(StateFilter(FMSPredict.get_user_data))
async def predict(message: Message, state: FSMContext):
    factor_values = message.text.strip().split()
    if len(factor_values) != 11:
       await message.reply("❌ Будь ласка, введіть рівно 11 чисел у правильному порядку.")
       return

    try:
        # Обробка даних
        factor_values = [factor_values]
        factor_values = scaler.transform(factor_values)
        factor_values = torch.tensor(factor_values, dtype=torch.float32)    
        
        # Прогнозування
        with torch.no_grad():
            out_logits = model(factor_values)

        # визначаємо ймовірність: щоб продемонструвати користувачу
        # перетворюємо логіти в ймовірності
        probabilities = torch.softmax(out_logits, dim=1)
        
        # визначаємо найбільшу ймовірність та її індекс  
        # індекс і буде класом: 0 - здоровий, 1-хворий
        class_probability, class_number = torch.max(probabilities, dim=1)

        # Інтерпретація результату
        if class_number.item() == 1:
            label = "⚠️ Є ризик серцево-судинного захворювання"
            await message.reply(f"{label}\nЙмовірність хвороби: {class_probability.item():.2f}")
        else:
            label = "✅ Без ознак захворювання"
            await message.reply(f"{label}\nЙмовірність хвороби: {1 - class_probability.item():.2f}")

    except Exception as e:
        await message.reply(f"Помилка обробки даних: {e}")
        await state.clear()()
    await state.set_state(default_state)


@user_router.message(StateFilter(default_state))
async def send_answer(message: Message):
    await message.answer(text="Ви ввели некоректну команду. Будь ласка, оберіть потрібну команду зі списку.")

