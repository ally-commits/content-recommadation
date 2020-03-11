import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv("Market_Basket_Optimisation.csv",header=None)

transaction = []

for i in range(0,7501):
	transaction.append([str(ds.values[i,j]) for j in range(0,20)])

#training the apriori usin/g the dataset
	
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)