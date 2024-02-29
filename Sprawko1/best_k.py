# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('stroke.csv', sep=';')
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
x = df.iloc[:, [1, 8]].values
y = df.iloc[:, 10].values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Przykładowe dane treningowe i etykiety
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)

# Definicja siatki parametrów do przeszukania
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]}

# Utworzenie modelu
knn = KNeighborsClassifier()

# Użycie GridSearchCV do znalezienia optymalnej wartości k
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Optymalna wartość k
best_k = grid_search.best_params_['n_neighbors']
print(best_k)