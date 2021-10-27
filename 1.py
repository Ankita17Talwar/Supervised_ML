'''
Data : house-votes-84.csv
'''
from sklearn import datasets
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

votes_df = pd.read_csv('data/house-votes-84.csv')
print(votes_df.shape)

# print(votes_df.info())
# print(votes_df.head())
# print(votes_df.describe())