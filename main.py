import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("Train.csv",)#  nrows = 10)
train.info()
# for col in train.columns: 
# 	print col+" : "+ str(train[col].nunique())

df_train = train.copy()
df_train['Outlet_Establishment_Year'] = 2019 - df_train['Outlet_Establishment_Year']

for col in train.columns:
	df_train[col].fillna(0, inplace = True) 

df_train = pd.get_dummies(df_train, columns=['Item_Identifier',	'Item_Fat_Content',	'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], drop_first=True)
df_train.head()
df_train.info()


