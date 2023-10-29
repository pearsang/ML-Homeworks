import numpy as np
import math
from scipy.stats import multivariate_normal


# EXERCISE 1

# Observations
x1 = np.array([[1], [0.6], [0.1]])
x2 = np.array([[0], [-0.4], [0.8]])
x3 = np.array([[0], [0.2], [0.5]])
x4 = np.array([[1], [0.4], [-0.1]])

X = np.array([x1, x2, x3, x4])
# 
pi1 = 0.5
pi2 = 0.5

# Bernoulli distribution - y1 
p1 = 0.3        # for y1 = 1 cluster 1
p2 = 0.7        # for y1 = 1 cluster 2



# Normal distribution - y2, y3
u1 = np.array([[1], [1]])
u2 = np.array([[0], [0]])
cov_matrix1 = np.array([[2, 0.5], [0.5, 2]])
cov_matrix2 = np.array([[1.5, 1], [1, 1.5]])


# Gaussian distribution function
def gaussianDistribution(x, u, cov_matrix):
    det = np.linalg.det(cov_matrix)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return (1/(2*np.pi))*(1/np.sqrt(det))*np.exp(-0.5 * (x - u).transpose().dot(inv_cov_matrix).dot(x - u))

# E-STEP - expectation step

# probability of x1 given that cluster 1: p(x1| c =1)
p_x1_c1 = p1 * gaussianDistribution(x1[1:], u1, cov_matrix1)

# probability of x1 given that cluster 2: p(x1| c =2)
p_x1_c2 = p2 * gaussianDistribution(x1[1:], u2, cov_matrix2)

# probability of x2 given that cluster 1: p(x2| c =1)
p_x2_c1 = (1-p1) * gaussianDistribution(x2[1:], u1, cov_matrix1)

# probability of x2 given that cluster 2: p(x2| c =2)
p_x2_c2 = (1-p2) * gaussianDistribution(x2[1:], u2, cov_matrix2)

# probability of x3 given that cluster 1: p(x3| c =1)
p_x3_c1 = (1-p1) * gaussianDistribution(x3[1:], u1, cov_matrix1)

# probability of x3 given that cluster 2: p(x3| c =2)
p_x3_c2 = (1-p2) * gaussianDistribution(x3[1:], u2, cov_matrix2)


# probability of x4 given that cluster 1: p(x4| c =1)
p_x4_c1 = p1 * gaussianDistribution(x4[1:], u1, cov_matrix1)

# probability of x4 given that cluster 2: p(x4| c =2)
p_x4_c2 = p2 * gaussianDistribution(x4[1:], u2, cov_matrix2)

# joint probability - ynk = p(c = k, xn)
y1_1 = pi1 * p_x1_c1
y1_2 = pi2 * p_x1_c2

y11 = y1_1 / (y1_1 + y1_2)
y12 = y1_2 / (y1_1 + y1_2)

y2_1 = pi1 * p_x2_c1
y2_2 = pi2 * p_x2_c2

y21 = y2_1 / (y2_1 + y2_2)
y22 = y2_2 / (y2_1 + y2_2)

y3_1 = pi1 * p_x3_c1
y3_2 = pi2 * p_x3_c2

y31 = y3_1 / (y3_1 + y3_2)
y32 = y3_2 / (y3_1 + y3_2)

y4_1 = pi1 * p_x4_c1
y4_2 = pi2 * p_x4_c2

y41 = y4_1 / (y4_1 + y4_2)
y42 = y4_2 / (y4_1 + y4_2)

# M-STEP - maximization step

N1 = y11 + y21 + y31 + y41
N2 = y12 + y22 + y32 + y42

print("n1: ", N1)
print("n2: ", N2)
print("x[1:]: ", x1[1:])
print("y11: ", y11)
print("y21: ", y21)
print("y31: ", y31)
print("y41: ", y41)


u1 = (1/N1) * (y11 * x1[1:] + y21 * x2[1:] + y31 * x3[1:] + y41 * x4[1:])
u2 = (1/N2) * (y12 * x1[1:] + y22 * x2[1:] + y32 * x3[1:] + y42 * x4[1:])

cov_matrix1 = (1/N1) * (y11 * (x1[1:] - u1).dot((x1[1:] - u1).transpose()) + y21 * (x2[1:] - u1).dot((x2[1:] - u1).transpose()) + y31 * (x3[1:] - u1).dot((x3[1:] - u1).transpose()) + y41 * (x4[1:] - u1).dot((x4[1:] - u1).transpose()))
cov_matrix2 = (1/N2) * (y12 * (x1[1:] - u2).dot((x1[1:] - u2).transpose()) + y22 * (x2[1:] - u2).dot((x2[1:] - u2).transpose()) + y32 * (x3[1:] - u2).dot((x3[1:] - u2).transpose()) + y42 * (x4[1:] - u2).dot((x4[1:]- u2).transpose()))

pi1 = N1 / (N1 + N2)
pi2 = N2 / (N1 + N2)

p1_new = (y11 * x1[0] + y21 * x2[0] + y31 * x3[0] + y41 * x4[0]) / N1
p2_new = (y12 * x1[0] + y22 * x2[0] + y32 * x3[0] + y42 * x4[0]) / N2

print("Updated Values:")
print("u1:\n", u1)
print("u2:\n", u2)
print("cov_matrix1:\n", cov_matrix1)
print("cov_matrix2:\n", cov_matrix2)
print("pi1:\n", pi1)
print("pi2:\n", pi2)
print("p1_new:\n", p1_new)
print("p2_new:\n", p2_new)



## EXERCISE 2

# New observation
x_new = np.array([[1], [0.3], [0.7]])

# Initialize an array to store the posteriors
posteriors = np.zeros(2)

# cluster 1
p_x_new_c1 = p1_new * gaussianDistribution(x_new[1:], u1, cov_matrix1)
print("p_x_new_c1: ", p_x_new_c1)
# cluster 2
p_x_new_c2 = p2_new * gaussianDistribution(x_new[1:], u2, cov_matrix2)
print("p_x_new_c2: ", p_x_new_c2)

# joint probability - ynk = p(c = k, xn)
y_new1 = pi1 * p_x_new_c1
y_new2 = pi2 * p_x_new_c2

ynew1 = y_new1 / (y_new1 + y_new2)
ynew2 = y_new2 / (y_new1 + y_new2)

# Print the posteriors
print("Cluster Memberships (Posteriors) for the New Observation:")
print("Cluster 1:", ynew1)
print("Cluster 2:", ynew2)


# EXERCISE 3
#p(xn|c=1)

p_x1_c1 = p1_new * gaussianDistribution(x1[1:], u1, cov_matrix1)
p_x2_c1 = (1 - p1_new) * gaussianDistribution(x2[1:], u1, cov_matrix1)
p_x3_c1 = (1 - p1_new) * gaussianDistribution(x3[1:], u1, cov_matrix1)
p_x4_c1 = p1_new * gaussianDistribution(x4[1:], u1, cov_matrix1)

#p(xn|c=2)

p_x1_c2 = p2_new * gaussianDistribution(x1[1:], u2, cov_matrix2)
p_x2_c2 = (1 - p2_new) * gaussianDistribution(x2[1:], u2, cov_matrix2)
p_x3_c2 = (1 - p2_new) * gaussianDistribution(x3[1:], u2, cov_matrix2)
p_x4_c2 = p2_new* gaussianDistribution(x4[1:], u2, cov_matrix2)



def find_belongs_to_cluster(p_x_c1, p_x_c2, x):
    print(x, "belongs to cluster 1" if p_x_c1 > p_x_c2 else "belongs to cluster 2")
        
find_belongs_to_cluster(p_x1_c1, p_x1_c2, "x1")
find_belongs_to_cluster(p_x2_c1, p_x2_c2, "x2")
find_belongs_to_cluster(p_x3_c1, p_x3_c2, "x3")
find_belongs_to_cluster(p_x4_c1, p_x4_c2, "x4")
    


def manhattan_distance(x, y):
    return np.abs(x[0][0]-y[0][0]) + np.abs(x[1][0]-y[1][0]) + np.abs(x[2][0]-y[2][0])

# a = medium distance from xi to the points in the same cluster
# minimum medium distance from x1 to the points of another cluster - in this case we only have two clusters

a_x1