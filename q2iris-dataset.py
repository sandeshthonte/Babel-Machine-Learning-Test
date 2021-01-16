import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics 

#a. loading iris dataset from csv file, printing the shape of the data, type of the data and first 20 rows
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
iris_dataset = pd.read_csv('D:\Iris.csv')
iris = iris_dataset.head(20)

print(iris_dataset.values)
print(iris_dataset.shape)
print(iris_dataset.dtypes)
print(iris)

#b. Print the number of NaN & missing values in the dataset – find the total in the dataset alongside the total in each column in the dataset.
print(iris_dataset.isnull().sum().sum())

X = iris_dataset.iloc[:, 1:-1].values
Y = iris_dataset.iloc[:, 5].values


#c.Splitting dataset in 80% test dataset, 20% train dataset, fitting dataset into model using the K Nearest Neighbor Algorithm.
#data,target = iris_dataset.load_iris(return_X_y=True)
x_train, x_test, y_train , y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#print(iris_dataset.info())
#print(X)
#print(Y)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
    
#-------------------------------------------------------------------
#tried this to understand how k can affect the score accuracy and selected k as 5
'''for i in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    y_predi = knn.predict(x_test)
    print(metrics.accuracy_score(y_test, y_predi))
    #print('Testing Accuracy:', str(knn.score(x_test, y_test)))
    #print('Training Accuracy:', str(knn.score(x_train, y_train))) 
'''

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_predi = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_predi))
print('Testing Accuracy:', str(knn.score(x_test, y_test)))
print('Training Accuracy:', str(knn.score(x_train, y_train)))

x_new = [[5.1,3.8,1.9,0.4], [5,2.3,3.3,1], [5,3.4,1.6,0.4]]
y_predi = knn.predict(x_new)
print(y_predi[0])
print(y_predi[1])
print(y_predi[2])
    
