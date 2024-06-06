import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
data=pd.read_csv('data.csv')
data.head()


#visualization

sns.FacetGrid(data, hue='Species')\
.map(plt.scatter,'SepalLengthCm','SepalWidthCm')\
.add_legend()
# plt.show()

sns.pairplot(data,hue='Species')
# plt.show()

#prepare train and test
x=data.iloc[:,:-1].values #feature variable
y=data.iloc[:,-1].values # targets

#splitting the train and test
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=0)

#decision tree
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

## summary of prediction made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

## accuracy score
print('acccuracy is',accuracy_score(y_pred, y_test))