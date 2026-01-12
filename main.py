import os
import json
import asyncio
import sqlite3
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from dotenv import load_dotenv
from groq import Groq
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

load_dotenv()
app = FastAPI()

# Використовуємо DefaultBotProperties для сумісності з новим aiogram
bot = Bot(
    token=os.getenv("TELEGRAM_BOT_TOKEN"),
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Завантаження моделі XGBoost
model = xgb.XGBClassifier()
model.load_model("loan_model.json")


# --- РОБОТА З БД ---
def init_db():
    conn = sqlite3.connect("bot_memory.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS history (user_id INTEGER PRIMARY KEY, data TEXT)")
    conn.commit()
    conn.close()


def get_user_history(user_id):
    conn = sqlite3.connect("bot_memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM history WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return json.loads(row[0]) if row else []


def save_user_history(user_id, history):
    conn = sqlite3.connect("bot_memory.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO history VALUES (?, ?)", (user_id, json.dumps(history)))
    conn.commit()
    conn.close()


init_db()


# --- КНОПКА ---
def get_clear_kb():
    builder = InlineKeyboardBuilder()
    builder.row(types.InlineKeyboardButton(text="🧹 Очистити діалог", callback_data="clear_history"))
    return builder.as_markup()


# --- ПРОГНОЗ ---
def get_prediction(data):
    df = pd.DataFrame([data])
    df = df[['age', 'income', 'loan_amount', 'credit_score']]
    prob = model.predict_proba(df)[0][1]
    status = "Схвалено" if model.predict(df)[0] == 1 else "Відхилено"
    return status, f"{round(float(prob) * 100, 2)}%"


# --- ОБРОБНИКИ ---
@dp.message(Command("start"))
async def start(message: types.Message):
    save_user_history(message.from_user.id, [])
    await message.answer(
        "<b>Вітаю! Я ваш кредитний асистент.</b> 🏦\n\n"
        "Напишіть мені ваш вік, дохід за місяць, бажану суму кредиту та кредитний рейтинг.",
        reply_markup=get_clear_kb()
    )


@dp.callback_query(F.data == "clear_history")
async def clear_history_handler(callback: types.CallbackQuery):  # Правильна назва типу
    save_user_history(callback.from_user.id, [])
    await callback.answer("Дані видалено")
    await callback.message.answer("Контекст очищено. Я готовий до нових розрахунків!", reply_markup=get_clear_kb())


@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    history = get_user_history(user_id)
    history.append({"role": "user", "content": message.text})

    try:
        # 1. Витягуємо дані (Використовуємо 70b модель для точності)
        extract = client.chat.completions.create(
            messages=[{"role": "system",
                       "content": "Extract: age, income, loan_amount, credit_score. Return ONLY JSON. If missing, use null."}] + history,
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        data = json.loads(extract.choices[0].message.content)

        required = ["age", "income", "loan_amount", "credit_score"]
        missing = [f for f in required if not data.get(f)]

        if not missing:
            # 2. Розрахунок
            status, prob = get_prediction(data)

            # СУВОРИЙ ПРОМПТ ДЛЯ ВІДПОВІДІ (запобігаємо галюцинаціям)
            prompt = (f"Ти менеджер українського банку. Клієнту {status} кредит з імовірністю {prob}. "
                      "Напиши коротке, ввічливе рішення українською мовою. "
                      "НЕ вигадуй про чеські ринки, дитинство чи інвестиції. "
                      "НЕ використовуй дужки [] або плейсхолдери. Пиши як людина.")

            res = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            bot_answer = res.choices[0].message.content
        else:
            # 3. Запит відсутніх даних
            prompt = (f"Нам бракує: {missing}. Попроси клієнта надати ці дані природною українською мовою. "
                      "Будь лаконічним. Жодних списків зі зірочками **.")
            res = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            bot_answer = res.choices[0].message.content

        history.append({"role": "assistant", "content": bot_answer})
        save_user_history(user_id, history[-6:])
        await message.answer(bot_answer, reply_markup=get_clear_kb())

    except Exception as e:
        print(f"Error: {e}")
        await message.answer("Вибачте, технічна помилка. Спробуйте уточнити дані.", reply_markup=get_clear_kb())


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(dp.start_polling(bot))


@app.get("/")
def home(): return {"status": "online"}