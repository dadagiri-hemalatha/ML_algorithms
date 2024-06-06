import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
data=pd.read_csv('data.csv')
data.head()
# print(data.shape)
# print(data.info)


#visualization

sns.FacetGrid(data, hue='Species')\
.map(plt.scatter,'SepalLengthCm','SepalWidthCm')\
.add_legend()
# plt.show()

sns.pairplot(data,hue='Species')
# plt.show()

#prepare train and test
x=data.iloc[:,:-1].values #feature variable
y=data.iloc[:,:-1].values # targets

#splitting the train and test
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=0)

#linear model **Y=a*X+b
#converting objects data type into int data type using Label Encoder
xl=data.iloc[:,:-1].values
yl=data.iloc[:,-1].values

le=LabelEncoder()
y_train=le.fit_transform(yl)
# print(y_train) #y_train categorical to numerical

## only for linear regression

x_trainl, x_testl, y_trainl, y_testl=train_test_split(xl,y_train, test_size=0.3, random_state=0)

model=LinearRegression()
model.fit(x_trainl,y_trainl)
y_pred=model.predict(x_testl)

# calculating the residuals
print('y-intercept             :' , model.intercept_)
print('beta coefficients       :' , model.coef_)
print('Mean Abs Error MAE      :' ,metrics.mean_absolute_error(y_testl,y_pred))
print('Mean Sqrt Error MSE     :' ,metrics.mean_squared_error(y_testl,y_pred))
print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_testl,y_pred)))
print('r2 value                :' ,metrics.r2_score(y_testl,y_pred))