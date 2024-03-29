#data_preprocessing

# 1. Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 2. Import the datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values



# 3. Splitting the dataset into testing and training dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state=0)


# 4. Fitting the linear regression model to training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# 5. Predicting the Test set results
y_pred = regressor.predict(X_test)

# 6. Visualize the training set results
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# 7. Visualize the testing set results
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience(Testing Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



                                
                                                      