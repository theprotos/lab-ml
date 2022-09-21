import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

dataset = "dataset\housing.csv"
print(dataset)
df = pd.read_csv(dataset)
print(df.columns)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(4, 4))
ax[0].title.set_text('crim')
ax[0].hist(df['crim'])
ax[1].title.set_text('rm')
ax[1].hist(df['rm'])
ax[2].scatter(df['crim'], df['medv'])
ax[3].scatter(df['rm'], df['medv'])
plt.show()


# Split data to teach and train
X, y = df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
           'ptratio', 'b', 'lstat']], df['medv']

print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()

model.fit(X_train, y_train)

print(mse(y_train, model.predict(X_train)))