import os

from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def predictor(request):
    return render(request, 'index.html')


def result(request):
    csv_directory = os.getcwd() + '\DiabetesPredictor\diabetes.csv'
    data = pd.read_csv(csv_directory)
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = LogisticRegression(max_iter=2000)
    model.fit(x_train.values, y_train.values)

    val1 = float(request.GET['n1'].strip() or 0)
    val2 = float(request.GET['n2'].strip() or 0)
    val3 = float(request.GET['n3'].strip() or 0)
    val4 = float(request.GET['n4'].strip() or 0)
    val5 = float(request.GET['n5'].strip() or 0)
    val6 = float(request.GET['n6'].strip() or 0)
    val7 = float(request.GET['n7'].strip() or 0)
    val8 = float(request.GET['n8'].strip() or 0)

    prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    if prediction == [1]:
        result1 = "We are sorry but you might have diabetes!!!"
    else:
        result1 = "Congratulations, you dont have diabetes!!!"

    return render(request, 'index.html', {"result2": result1})
