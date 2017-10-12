import numpy as np

np.random.seed(1)

def initializeParams(nx, nh, ny):
  W1 = np.random.randn(nh, nx)
  b1 = np.zeros((nh, 1))
  W2 = np.random.randn(ny, nh)
  b2 = np.zeros((ny, 1))

  parameters = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

  return parameters

def tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def sigmoid(x):
  return 1./(1+np.exp(-x))

def sigmoidprime(x):
  return sigmoid(x)*(1-sigmoid(x))

def tanhprime(x):
  return 1 - tanh(x)*tanh(x)

def feedForward(X, parameters):
  W1=parameters['W1']
  b1=parameters['b1']
  W2=parameters['W2']
  b2=parameters['b2']

  Z1 = np.dot(W1, X) + b1
  A1 = tanh(Z1)
  Z2 = np.dot(W2, A1) + b2
  A2 = sigmoid(Z2)

  cache = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'Z1':Z1, 'Z2':Z2, 'A1':A1, 'A2':A2}

  return A2, cache

def computeLoss(Y, yhat):
  m = Y.shape[1]
  loss = 1./m*np.sum(np.multiply(Y, np.log(yhat)) + np.multiply(1-Y, np.log(1-yhat)))
  return -loss


def backProp(X, Y, yhat, cache):
  m = Y.shape[1]

  dA2 = (-Y/yhat + (1-Y)/(1-yhat))
  dZ2 = np.multiply(dA2, sigmoidprime(cache['Z2']))
  dW2 = 1./m*np.dot(dZ2, cache['A1'].T)
  db2 = 1./m*np.sum(dZ2, axis=1, keepdims=True)
  dA1 = np.dot(cache['W2'].T, dZ2)
  dZ1 = np.multiply(dA1, sigmoidprime(cache['Z1']))
  dW1 = 1./m*np.dot(dZ1, X.T)
  db1 = 1./m*np.sum(dZ1, axis=1, keepdims=True)

  dvar = {'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2, 'dZ1':dZ1, 'dZ2':dZ2, 'dA1':dA1, 'dA2':dA2}

  return dvar

def updateParams(parameters, dparams, alpha):
  parameters['W1'] = parameters['W1'] - alpha*dparams['dW1']
  parameters['b1'] = parameters['b1'] - alpha*dparams['db1']
  parameters['W2'] = parameters['W2'] - alpha*dparams['dW2']
  parameters['b2'] = parameters['b2'] - alpha*dparams['db2']

  return parameters


def performNNfit(X, Y, nh, alpha, epochs):
  nx = X.shape[0]
  ny = Y.shape[0]
  #initialize parameters
  params = initializeParams(nx, nh, ny)
  outputloss = []

  for i in range(epochs):
    yhat, cache = feedForward(X, params)
    loss = computeLoss(Y, yhat)
    dvar = backProp(X, Y, yhat, cache)
    params = updateParams(params, dvar, alpha)
    outputloss.append(loss)

  return params, outputloss

def performNNpredict(X, params):
  yhat, cache = feedForward(X, params)
  yhat[np.where(yhat<0.5)]=0
  yhat[np.where(yhat>=0.5)]=1

  return yhat


