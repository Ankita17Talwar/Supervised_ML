import pandas as pd


boston = pd.read_csv('data/boston.csv')

print(boston.head())

# target variable = MEDV - median value
# Create Featue and Target

X = boston.drop('MEDV', axis=1).values
y= boston['MEDV'].values