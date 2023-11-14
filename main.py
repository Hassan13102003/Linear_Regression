import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris=pd.read_csv('iris.csv')
print(iris.head())

y=iris[['SepalLengthCm']]
x=iris[['PetalLengthCm']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
print(y_test.head())
print(y_pred[0:5])

print(mean_squared_error(y_test,y_pred))





