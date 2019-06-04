import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Import the dataset
dataset = pandas.read_csv('data_2.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into the training set and test set
# We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,
# and 10 rows will go into the testing set.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(xTrain, yTrain)

# Make predictions using the testing set
yPred = regr.predict(xTest)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(yTest, yPred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(yTest, yPred))

# Plot outputs
plt.scatter(xTest, yTest,  color='black')
plt.plot(xTest, yPred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()