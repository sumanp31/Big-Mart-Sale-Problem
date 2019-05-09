import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv("Train.csv", nrows = 2000)
df.head()
df.info()

df_train = df.copy()
df_train = df_train.iloc[:, 0:11]

df_train['Outlet_Establishment_Year'] = 2019 - df_train['Outlet_Establishment_Year']

for col in df_train.columns:
	df_train[col].fillna(0, inplace = True) 
df_train.info()

for col in df_train.columns:
	sns.catplot(col, 'Item_Outlet_Sales',data= df)

plt.show()