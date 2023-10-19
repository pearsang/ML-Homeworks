import numpy as np

# EXERCISE 2
x1 = np.matrix([[1], [1], [1], [1]])
x2 = np.matrix([[1], [0], [0], [-1]])

t1 = np.matrix([[1], [0], [0]])
t2 = np.matrix([[0], [1], [0]])

W1 = np.matrix([[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]])
W2 = np.matrix([[1, 4, 1], [1, 1, 1]])
W3 = np.matrix([[1, 1], [3, 1], [1, 1]])

b1 = np.matrix([[1], [1], [1]])
b2 = np.matrix([[1], [1]])
b3 = np.matrix([[1], [1], [1]])

def activationFunction(x):
    return np.tanh(0.5* x - 2)

def derivativeActivationFunction(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = (2/(np.exp(0.5*x[i][j] - 2) + np.exp(-0.5*x[i][j] + 2))**2)
    return x


# Calculations for x1
a1_1 = W1.dot(x1) + b1
v1_1 = activationFunction(a1_1)

print(derivativeActivationFunction(a1_1))

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
temp = np.subtract(t1, v3_1)
d3_1 = np.multiply(derivativeActivationFunction(a3_1), temp)
d2_1 = np.multiply(derivativeActivationFunction(a2_1), W3.transpose().dot(d3_1))
d1_1 = np.multiply(derivativeActivationFunction(a1_1), W2.transpose().dot(d2_1))
print(W2.transpose().dot(d2_1))
print(derivativeActivationFunction(a1_1))
print("D3_1: \n", d3_1)
print("D2_1: \n", d2_1)
print("D1_1: \n", d1_1)


# for x2
temp = np.subtract(t2, v3_2)
d3_2 = np.multiply(derivativeActivationFunction(a3_2), temp)
d2_2 = np.multiply(derivativeActivationFunction(a2_2), W3.transpose().dot(d3_2))
d1_2 = np.multiply(derivativeActivationFunction(a1_2), W2.transpose().dot(d2_2))

# print fisrt element of d1_1 with 20 decimal places
print("D1_1: \n", d1_1[0].round(20))

