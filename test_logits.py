import torch
import torch.nn as nn
import torch.nn.functional as F

import joblib

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


# Завантажуємо scaler та модель
scaler = joblib.load("scaler.pkl")
model = Model(input_factors=11)
model.load_state_dict(torch.load("cardio_quickstart_mlp.pth", map_location=torch.device("cpu")))
model.eval()


# Прогнозуємо
factor_values = '55 1 180 75 120 80 2 1 0 0 1'

factor_values = factor_values.strip().split()
print(factor_values)

factor_values = [factor_values]
print(factor_values)

factor_values = scaler.transform(factor_values)
print(factor_values)

factor_values = torch.tensor(factor_values, dtype=torch.float32)
print(factor_values)


with torch.no_grad():
    out_logits = model(factor_values)
    
    # визначаємо ймовірність - просто щоб показати користувачу
    # перетворюємо логіти в ймовірності
    probabilities = torch.softmax(out_logits, dim=1)

    # визначаємо найбільшу ймовірність і її індекс  # індекс і буде клас: 0 - немає або 1-є
    class_probability, class_number = torch.max(probabilities, dim=1)

print('NN output =', out_logits)
print('NN output type =', type(out_logits))

print('logits to probs =', probabilities)
print('predicted_class =', class_number.item())
print('class_prob =', class_probability.item())