import numpy as np
import math

# EXERCISE 2
learning_rate = 0.1

x1 = np.array([[1], [1], [1], [1]])
x2 = np.array([[1], [0], [0], [-1]])

t1 = np.array([[0], [1], [0]])
t2 = np.array([[1], [0], [0]])

W1 = np.array([[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]])
W2 = np.array([[1, 4, 1], [1, 1, 1]])
W3 = np.array([[1, 1], [3, 1], [1, 1]])

b1 = np.array([[1], [1], [1]])
b2 = np.array([[1], [1]])
b3 = np.array([[1], [1], [1]])

def activationFunction(x):
    return np.tanh(0.5* x - 2)

def derivativeActivationFunction(x):
    return 0.5 / pow((np.cosh(2-(0.5*x))), 2)


# Calculations for x1
a1_1 = W1.dot(x1) + b1
v1_1 = activationFunction(a1_1)

a2_1 = W2.dot(v1_1) + b2
v2_1 = activationFunction(a2_1)

a3_1 = W3.dot(v2_1) + b3
v3_1 = activationFunction(a3_1)


# Calculations for x2
a1_2 = W1.dot(x2) + b1
v1_2 = activationFunction(a1_2)

a2_2 = W2.dot(v1_2) + b2
v2_2 = activationFunction(a2_2)

a3_2 = W3.dot(v2_2) + b3
v3_2 = activationFunction(a3_2)

####### Backpropagation #######
# Derivatives - hadamard product

# for x1
temp = np.subtract(v3_1, t1)
d3_1 = np.multiply(derivativeActivationFunction(a3_1), temp)
d2_1 = np.multiply(derivativeActivationFunction(a2_1), W3.transpose().dot(d3_1))
d1_1 = np.multiply(derivativeActivationFunction(a1_1), W2.transpose().dot(d2_1))

# for x2
temp = np.subtract(v3_2, t2)
d3_2 = np.multiply(derivativeActivationFunction(a3_2), temp)
d2_2 = np.multiply(derivativeActivationFunction(a2_2), W3.transpose().dot(d3_2))
d1_2 = np.multiply(derivativeActivationFunction(a1_2), W2.transpose().dot(d2_2))


## update weigths
w3_update = np.subtract(W3, learning_rate * (d3_1.dot(v2_1.transpose()) + d3_2.dot(v2_2.transpose())))
w2_update = np.subtract(W2, learning_rate * (d2_1.dot(v1_1.transpose()) + d2_2.dot(v1_2.transpose())))
w1_update = np.subtract(W1, learning_rate * (d1_1.dot(x1.transpose()) + d1_2.dot(x2.transpose())))


print("WEIGHTS")
print(w1_update)
print(w2_update)
print(w3_update)


b1_update = np.subtract(b1, learning_rate * (d1_1 + d1_2))
b2_update = np.subtract(b2, learning_rate * (d2_1 + d2_2))
b3_update = np.subtract(b3, learning_rate * (d3_1 + d3_2))

print("BIASES")
print(b1_update)
print(b2_update)
print(b3_update)
