import pandas as pd
import matplotlib.pyplot as plt

boston = pd.read_csv('data/boston.csv')

print(boston.head())

# target variable = MEDV - median value
# Create Featue and Target

X = boston.drop('MEDV', axis=1).values
y= boston['MEDV'].values

# Predicting house value from single feature : slice number of rooms column
X_rooms = X[:,5]
print(type(X_rooms))
print(type(y))

y= y.reshape(-1,1)
X_rooms = X_rooms.reshape(-1,1)

# Plot house value  vs number of rooms
plt.scatter(X_rooms,y)
plt.ylabel('Value of house /1000($)')
plt.xlabel('Number of rooms')
plt.show()

# Fitting regression model
from sklearn.linear_model import LinearRegression
import numpy as np

reg = LinearRegression()

reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()

# Lasso feature selection
from sklearn.linear_model import Lasso

names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation =60)
_ = plt.ylabel('Coefficients')
plt.show()