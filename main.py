import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode

warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print train.apply(lambda x: sum(x.isnull()))
print train.describe()
print train.apply(lambda x: len(x.unique()))



cate_col = [x for x in train.dtypes.index if train.dtypes[x] == 'object']
cate_col = [x for x in cate_col if x not in ['Item_Identifier', 'Outlet_Identifier']]

for col in cate_col:
    print '\nFrequency of Categories for varible %s'%col
    print train[col].value_counts()

## Data cleaning
df_train = train.copy()
df_test = test.copy()

df_train['source'] = 'train'
df_test['source'] = 'test'
df = pd.concat([df_train, df_test], ignore_index=True)
print df_train.shape, df_test.shape, df.shape

df['Outlet_age'] = 2013 - df['Outlet_Establishment_Year']


# fill missing weight with avg weight
miss_bool = df['Item_Weight'].isnull()
avg_weight = df.pivot_table(values='Item_Weight', index='Item_Identifier')
print '\nOrignal #missing: %d'% sum(miss_bool)
df.loc[miss_bool, 'Item_Weight'] = df.loc[miss_bool,
                                          'Item_Identifier'].apply(lambda x: avg_weight.loc[x])
ms = df['Item_Weight'].isnull()
print 'Final missing: %d'% sum(ms)

#fill missing outlet size with mode outlet size
miss_bool = df['Outlet_Size'].isnull()
outlet_size_mode = df.pivot_table(values='Outlet_Size',
                                  columns='Outlet_Type', aggfunc=(lambda x: mode(x).mode[0]))
print '\nOrignal #missing: %d'% sum(miss_bool)
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool,
                                          'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
ms = df['Outlet_Size'].isnull()
print 'Final missing: %d'% sum(ms)

#fill 0 in Item_Visibility
# ms = list(df[df['Item_Visibility'] == 0].index)
miss_bool = (df['Item_Visibility'] == 0)
avg_Visibility = df.pivot_table(values='Item_Visibility', index='Item_Identifier')
print 'Orignal missing: %d'% sum(ms)
df.loc[miss_bool, 'Item_Visibility'] = df.loc[miss_bool, 'Item_Identifier'].apply(lambda x: avg_Visibility.loc[x])

ms = (df['Item_Visibility'] == 0)
print 'Final missing: %d'% sum(ms)


# merge repeating categories in Item_fat_content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat',
	                                                        'reg':'Regular', 'low fat':'Low Fat'})

#Determine another variable with means ratio
for i in df.index:
    x = df.loc[i, 'Item_Identifier']
    df.loc[i, 'Item_Visibility_MeanRatio'] = df.loc[i,
                                                    'Item_Visibility']/float(avg_Visibility.loc[x])

print df['Item_Visibility_MeanRatio'].describe()

# Combine categories of Item_type

df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food',
	                                                        'NC':'Non-Consumable', 'DR':'Drinks'})
print df['Item_Type_Combined'].value_counts()


# Mark non-consumables as separate category in fat content:
df.loc[df['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
df['Item_Fat_Content'].value_counts()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size',
           'Item_Type', 'Outlet_Type', 'Outlet']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Item_Type_Combined', 'Outlet',
	                                'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

df.drop(['Outlet_Establishment_Year', 'Item_Visibility_MeanRatio',
	        'Item_Type'], axis=1, inplace=True)

#Divide into test and train:


df_train = df.loc[df['source'] == "train"]
df_test = df.loc[df['source'] == "test"]

data = pd.DataFrame(columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
data['Item_Identifier'] = df_test["Item_Identifier"]
data['Outlet_Identifier'] = df_test["Outlet_Identifier"]

#Drop unnecessary columns:
df_test.drop(['Item_Outlet_Sales', 'source', 'Item_Identifier',
	             'Outlet_Identifier'], axis=1, inplace=True)
df_train.drop(['source', 'Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)

#Export files as modified versions:
df_train.to_csv("Train_modified.csv", index=False)
df_test.to_csv("test_modified.csv", index=False)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Item_Outlet_Sales', axis=1),
	                                                   df_train['Item_Outlet_Sales'], test_size=0.01)


## Prediction

# from sklearn.linear_model import LinearRegression
# lreg = LinearRegression(normalize=True)
# lreg.fit(X_train,y_train)
# pred_lreg = lreg.predict(X_test)

# from sklearn.metrics import mean_squared_error
# print "RMSE for LinearRegression: %.4g" % np.sqrt(mean_squared_error(y_test, pred_lreg))

# # Predict on testing data:
# pred_test_lreg = lreg.predict(df_test)
# df_test.loc[:, 'Item_Outlet_Sales'] = pred_test_lreg
# data.loc[:, 'Item_Outlet_Sales'] = pred_test_lreg
# data.to_csv('submission_lreg.csv', index=False)


# from sklearn.tree import DecisionTreeRegressor
# dtreg = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
# dtreg.fit(X_train,y_train)
# pred_dtreg = dtreg.predict(X_test)

# from sklearn.metrics import mean_squared_error
# print "RMSE for Decison Tree: %.4g" % np.sqrt(mean_squared_error(y_test, pred_dtreg))

# # Predict on testing data:
# pred_test_dtreg = dtreg.predict(df_test)
# df_test.loc[:, 'Item_Outlet_Sales'] = pred_test_dtreg
# data.loc[:, 'Item_Outlet_Sales'] = pred_test_dtreg
# data.to_csv('submission_dtreg.csv', index=False)


from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
rfreg.fit(X_train, y_train)
pred_rfreg = rfreg.predict(X_test)

from sklearn.metrics import mean_squared_error
print "RMSE for Random Forest: %.4g" % np.sqrt(mean_squared_error(y_test, pred_rfreg))

# Predict on testing data:
pred_test_rfreg = rfreg.predict(df_test)
df_test.loc[:, 'Item_Outlet_Sales'] = pred_test_rfreg
data.loc[:, 'Item_Outlet_Sales'] = pred_test_rfreg
data.to_csv('submission_rfreg.csv', index=False)
