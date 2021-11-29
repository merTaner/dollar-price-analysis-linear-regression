import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

veri = pd.read_csv("Dolar-Fiyat-n-Tahmin-Edelim/2016dolaralis.csv")

x = veri["Gun"]
y = veri["Fiyat"]

x =  x.values.reshape(251,1)
y =  y.values.reshape(251,1)

# Lineer Reg.
tahminLineer = LinearRegression()
tahminLineer.fit(x,y)
tahminLineer.predict(x)

"""
print(float(y[0]))
print(float(tahminLineer.predict(x)[0]))
"""

# Polynomial Reg. 
tahminPolinom = PolynomialFeatures(degree=8)
xYeni = tahminPolinom.fit_transform(x)

polinomModel = LinearRegression()
polinomModel.fit(xYeni, y)
polinomModel.predict(xYeni)


plt.scatter(x, y)
plt.plot(x,tahminLineer.predict(x), color='red')
plt.plot(x, polinomModel.predict(xYeni), c='orange')
plt.show()

"""
print(float(y[0]))
print(float(polinomModel.predict(xYeni)[0]))


hatalarınKaresiLineer = 0
hatalarınKaresiPolinom = 0

for i in range(len(y)):
    hatalarınKaresiLineer = hatalarınKaresiLineer + (float(y[i]) - float(tahminLineer.predict(x)[i]))
    # print(hatalarınKaresiLineer)

#print("-"*50)
for i in range(len(xYeni)):
    hatalarınKaresiPolinom = hatalarınKaresiPolinom + (float(y[i]) - float(polinomModel.predict(xYeni)[i]))
    # print(hatalarınKaresiPolinom)

"""

hatalarınKaresiPolinom = 0
for a in range(10):
    tahminPolinom = PolynomialFeatures(degree = a+1)
    xYeni = tahminPolinom.fit_transform(x)

    polinomModel = LinearRegression()
    polinomModel.fit(xYeni, y)
    polinomModel.predict(xYeni)

    for i in range(len(xYeni)):
        hatalarınKaresiPolinom = hatalarınKaresiPolinom + (float(y[i]) - float(polinomModel.predict(xYeni)[i]))**2
    print(a+1, "inci dereceden fonksiyonun hatası", hatalarınKaresiPolinom)

    hatalarınKaresiPolinom = 0




