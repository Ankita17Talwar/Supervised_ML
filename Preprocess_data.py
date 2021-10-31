import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/gm_2008_region.csv')

print(df.columns)

# Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
#        'BMI_female', 'life', 'child_mortality', 'Region'],
#       dtype='object')

# Note :Region is categorical feature

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
#        'BMI_female', 'life', 'child_mortality', 'Region_America',
#        'Region_East Asia & Pacific', 'Region_Europe & Central Asia',
#        'Region_Middle East & North Africa', 'Region_South Asia',
#        'Region_Sub-Saharan Africa'],
#       dtype='object')


# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

# use ridge regression to perform 5-fold cross-validation.
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

X = df_region.drop('life', axis=1)
y = df['life']
y = y.reshape(-1, 1)

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)