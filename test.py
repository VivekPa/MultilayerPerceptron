import pandas as pd
import numpy as np
from MLP import MLP

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 1, 0)

x = np.asarray(df.iloc[0:100, [0, 2]].values)

x_test = np.asarray(df.iloc[2, [0, 2]].values)

# print(x)
mlp = MLP(0.2, x, y)
# print(mlp.output1)
# print('Initial weights: ')
# print(mlp.print_weights())

mlp.train()
# print('Trained Weights: ')
# print(mlp.print_weights())

# mlp.feedforward(x_test)
# print(mlp.z2)
# print(mlp.z2)
# print(mlp.output1)

# print(x.shape[1])
