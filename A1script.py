#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

csvfile = sys.argv[1]



df = pd.read_csv (csvfile)


x = df.x
y = df.y

plt.scatter(x,y)
plt.title('regrex1')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Png_scatter.png")    




x = df.x.to_numpy()
x = x.reshape(-1, 1)
y = df.y.to_numpy()
y = y.reshape(-1, 1)
reg = LinearRegression().fit(x, y)
reg.score(x, y)
reg.coef_
reg.intercept_
y_predict = reg.predict(x)




plt.scatter(x,y)
plt.plot(x,y_predict)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Png_Linear.png") 








