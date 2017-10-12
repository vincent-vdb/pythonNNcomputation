import neuralNetwork2Layers as NN
import numpy as np
import matplotlib.pyplot as plt

nx = 4
nh = 3
ny = 1
alpha = 0.1
epochs = 10000
m = 15
X = np.random.randn(nx, m)
Y = np.random.randn(1, m)

Y[np.where(Y<0.5)]=0
Y[np.where(Y>=0.5)]=1

params, loss = NN.performNNfit(X, Y, nh, alpha, epochs)

predict = NN.performNNpredict(X, params)

x = np.arange(epochs)
print(x)

print(loss)

plt.plot(x, loss)
plt.show()

