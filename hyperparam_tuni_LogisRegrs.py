import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



# Import Data
df = pd.read_csv('data/diabetes.csv')
print(df.columns)
print(df.head())

# Feature and Target Variable
X = df.drop('diabetes',axis=1).values
y = df['diabetes']


# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}  # dictionary of list

# Create the classifier: logreg
logreg = LogisticRegression(max_iter=400)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

