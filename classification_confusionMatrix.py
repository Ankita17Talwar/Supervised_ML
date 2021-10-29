import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

# load data
diabetes = pd.read_csv('data/diabetes.csv')

print(diabetes.head())

# Feature and Target
X = diabetes.drop('diabetes', axis=1).values
y = diabetes['diabetes'].values

print (y.shape) # (768,)
print(X.shape) # (768, 8)

y= y.reshape(-1,1)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42 , stratify=y)


