import collections
import seaborn as sns
import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt
import collections
# >> YOUR CODE HER
#
# E
a = pd.read_csv("dataset/yelp.csv")
b = pd.DataFrame(a.groupby("state")['stars'].mean())
aa = b["stars"].to_numpy()
print(aa)