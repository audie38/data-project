import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, silhouette_score


# Pandas DataFrame
dt = pd.DataFrame(columns=["Nama", "Mafa"])
# dt = dt._append({"Nama":"Ichigo", "Mafa":"NasGor"}, ignore_index=True)
# dt = dt._append({"Nama":"Kurosaki", "Mafa":"MiGor"}, ignore_index=True)
# print(dt)

data = {
    "Nama":["Ichigo", "Kurosaki"],
    "Mafa":["NasGor", "MiGor"]
}

dt = pd.DataFrame(data)
dt["Mifa"] = ["Kopi", "Es Teh"]
print(dt)

# Numpy DataFrame
arr1 = np.array([1, 0, 0, 1])
arr2 = np.array([0, 1, 1, 2])

print(arr1.reshape(2, 2))
print(arr1 + arr2)
print(np.dot(arr1.reshape(2, 2), arr2.reshape(2, 2)))
print(np.where(arr2 > 0, "Yes", "No"))

# Plot
array_1 = np.array(range(-10,11))
array_2 = np.array(list(map(lambda x : x**2, list(array_1))))

plt.plot(array_1, array_2)
plt.show()

plt.bar(['Nasgor', 'Bubur Ayam', 'Mie Instant'], [10, 9, 7])
plt.show()

random_array_1 = np.random.randint(1, 100, 100)
random_array_2 = np.random.randint(1, 100, 100)

plt.scatter(random_array_1, random_array_2)
plt.show()

# SKLearn
