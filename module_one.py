import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris = pd.read_csv('Iris.csv')

print(iris.head())

sepal_width_gt_4 = iris[iris['SepalWidthCm'] > 4]
print(sepal_width_gt_4)

petal_width_gt_1 = iris[iris['PetalWidthCm'] > 1]
print(petal_width_gt_1)

petal_width_gt_2 = iris[iris['PetalWidthCm'] > 2]
print(petal_width_gt_2)

sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=iris)
plt.title('Relationship between Sepal Length and Petal Length')
plt.show()

sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', hue='Species', data=iris)
plt.title('Sepal Length vs Petal Length with Species')
plt.show()

y = iris[['SepalLengthCm']]
x = iris[['SepalWidthCm']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print("Actual values:\n", y_test.head())
print("Predicted values:\n", y_pred[:5])

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
