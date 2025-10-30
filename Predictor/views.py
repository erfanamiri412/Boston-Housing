from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('BostonHousing.csv')

x = df.drop('medv', axis=1)
y = df['medv']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)


def index(request):
    prediction = None
    error_msg = None
    is_farsi = request.POST.get('lang') == 'fa'

    features = {
        "CRIM": (0, 100),
        "ZN": (0, 100),
        "INDUS": (0, 30),
        "CHAS": (0, 1),
        "NOX": (0, 1),
        "RM": (1, 10),
        "AGE": (0, 100),
        "DIS": (1, 15),
        "RAD": (1, 24),
        "TAX": (100, 800),
        "PTRATIO": (10, 30),
        "B": (0, 400),
        "LSTAT": (0, 40)
    }

    if request.method == "POST":
        try:
            input_values = []
            for key, (min_val, max_val) in features.items():
                val_str = request.POST.get(key, "0")
                try:
                    val = float(val_str)
                except ValueError:
                    val = min_val
                val = max(min_val, min(val, max_val))
                input_values.append(val)

            X_input = np.array(input_values).reshape(1, -1)
            prediction = round(model.predict(X_input)[0], 2)
        except Exception as e:
            error_msg = "خطا در پیش‌بینی: " + \
                str(e) if is_farsi else "Prediction error: " + str(e)

    form_data = {key: request.POST.get(key, "") for key in features.keys()}

    return render(request, 'index.html', {
        'prediction': prediction,
        'error_msg': error_msg,
        'form_data': form_data,
        'is_farsi': is_farsi
    })
