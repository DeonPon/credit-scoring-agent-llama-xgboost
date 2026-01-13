#  словник на початку файлу після імпортів
LABELS = {
    "age": "ваш вік",
    "income": "щомісячний дохід",
    "loan_amount": "суму кредиту",
    "credit_score": "кредитний рейтинг"
}


@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    history = get_user_history(user_id)
    history.append({"role": "user", "content": message.text})

    try:
        # Екстракція даних
        extract = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "Extract fields: age, income, loan_amount, credit_score. Return ONLY JSON. Use null for missing."
            }] + history,
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        data = json.loads(extract.choices[0].message.content)

        required = ["age", "income", "loan_amount", "credit_score"]
        missing = [f for f in required if not data.get(f)]

        if not missing:
            status, prob = get_prediction(data)


            prompt = (
                f"Ти — Андрій, ввічливий менеджер 'ПриватБанку'. Клієнту {status} кредит з імовірністю {prob}. "
                "Напиши офіційне, але тепле рішення українською мовою. "
                "Подякуй за довіру, поясни, що ми проаналізували дані. "
                "Уникай технічних символів, дужок або ієрогліфів. Тільки чистий текст."
            )

            res = client.chat.completions.create(
                messages=[{"role": "system", "content": "Ти професійний банківський клерк."},
                          {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7
            )
            bot_answer = res.choices[0].message.content
        else:
            #  ЛЮДЯНИЙ ЗАПИТ ДАНИХ
            missing_names = [LABELS[k] for k in missing]
            missing_text = ", ".join(missing_names)

            prompt_missing = (
                f"Ти менеджер банку. Нам бракує таких даних: {missing_text}. "
                "Попроси клієнта надати їх дуже ввічливо, одним реченням, як у живому чаті. "
                "Не використовуй списки."
            )

            res = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_missing}],
                model="llama-3.1-8b-instant"  # Швидка модель для простих питань
            )
            bot_answer = res.choices[0].message.content

        # Зберігаємо та відправляємо
        history.append({"role": "assistant", "content": bot_answer})
        save_user_history(user_id, history[-6:])
        await message.answer(bot_answer, reply_markup=get_clear_kb())

    except Exception as e:
        print(f"Error: {e}")
        await message.answer("Я трохи заплутався в цифрах. Напишіть, будь ласка, ще раз ваш дохід та рейтинг.")