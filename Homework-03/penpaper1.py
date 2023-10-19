import numpy as np

# EXERCISE 1

data = np.array([[0.7,-0.3],[0.4,0.5],[-0.2,0.8], [-0.4, 0.3]])

# initialize a matrix, 4x4 filled with 0s
phi = np.zeros((4,4))

targets = np.array([0.8, 0.6, 0.3, 0.3])
c = np.array([[0, 0], [1, -1], [-1, 1]])
lmbda = 0.1

# A)
# Calculate Φ for each data point
def radial_basis(x, c):
    return np.exp(-(np.linalg.norm(x - c) ** 2)/2)

print("Radial Basis Function")
k = 0
for i in range(4):
    for j in range(4):
        if(j==0):
            phi[i][j] = 1
        else:
            k = j - 1
            phi[i][j] = radial_basis(data[i], c[k])

phiT = np.transpose(phi)
phiTphi = np.matmul(phiT, phi)
pseudo_inverse = np.linalg.inv(phiTphi + lmbda * np.identity(4))
phyTtargets = phiT.dot(targets)
weights = pseudo_inverse.dot(phyTtargets)



# B)
prediction = phi.dot(weights)

#root mean square error
rmse = np.sqrt(np.mean((prediction - targets) ** 2))

print("Root Mean Square Error: ", rmse)