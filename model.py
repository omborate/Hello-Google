import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df= pd.read_csv('StartupVal.csv')
df.nunique(axis=0)

x = df.drop(['Acquired','Domain'],axis=1)
y= df['Acquired']

from sklearn.model_selection import train_test_split
x_train, x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)

##picke file
pickle.dump(classifier, open('model.pkl', "wb"))