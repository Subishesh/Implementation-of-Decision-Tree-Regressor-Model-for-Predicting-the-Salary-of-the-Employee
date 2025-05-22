# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

   1.Import the libraries and read the data frame using pandas.
   2.Calculate the null values present in the dataset and apply label encoder.
   3. Determine test and training data set and apply decison tree regression in dataset.
   4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SUBISHESH P
RegisterNumber: 21222320220


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

 
*/
```

## Output:
![image](https://github.com/user-attachments/assets/7dc67364-a97a-4bfb-96e7-5b76020a0b07)

![image](https://github.com/user-attachments/assets/734639be-0944-4f76-83e8-34fc1402d869)

![image](https://github.com/user-attachments/assets/0a44a29e-da0d-4743-aacd-900703ac4799)

![image](https://github.com/user-attachments/assets/aaee0ce2-03d3-460f-8061-491553dfea17)

![image](https://github.com/user-attachments/assets/14087411-234b-4fbb-8f7d-a6f5449b4ac0)

![image](https://github.com/user-attachments/assets/8a88bbef-38d2-4c4d-a6c0-b768f59ced75)

![image](https://github.com/user-attachments/assets/e5e94585-5829-4092-9dc9-6308b6c0c1f9)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
