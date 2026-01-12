import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Створюємо маленьку базу даних для навчання
data = {
    'age': [25, 45, 30, 50, 22, 35],
    'income': [50000, 80000, 20000, 120000, 15000, 60000],
    'loan_amount': [10000, 20000, 50000, 10000, 5000, 15000],
    'credit_score': [700, 750, 400, 800, 350, 650],
    'is_approved': [1, 1, 0, 1, 0, 1]  # 1 - схвалено, 0 - відмова
}

df = pd.DataFrame(data)
X = df.drop('is_approved', axis=1)
y = df['is_approved']

# 2. Навчаємо XGBoost
model = xgb.XGBClassifier()
model.fit(X, y)

# 3. ЗБЕРІГАЄМО результат у файл
model.get_booster().save_model("loan_model.json")
print("Файл loan_model.json успішно наповнений мізками моделі!")