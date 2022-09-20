import pandas as pd
import matplotlib.pyplot as plt

dataset = "dataset\housing.csv"
print(dataset)
df = pd.read_csv(dataset)
print(df.head())

plt.hist(df['crim'])
plt.show()
