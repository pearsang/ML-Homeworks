import numpy as np
from scipy.stats import multivariate_normal

# Given parameters
pi = np.array([0.5, 0.5])
p = np.array([0.3, 0.7])
u = np.array([[1, 1], [0, 0]])
sigma = np.array([[[2, 0.5], [0.5, 2]], [[1.5, 1], [1, 1.5]]])

# Observations
observations = np.array([[1, 0.6, 0.1], [0, -0.4, 0.8], [0, 0.2, 0.5], [1, 0.4, -0.1]])

# Initialize variables to store responsibilities
responsibilities = np.zeros((len(observations), 2))

# E-Step
for i in range(len(observations)):
    for j in range(2):
        # Calculate the likelihood of the observation under the j-th cluster
        bernoulli_likelihood = p[j] ** observations[i, 0] * (1 - p[j]) ** (1 - observations[i, 0])
        mvn_likelihood = multivariate_normal.pdf(observations[i, 1:], mean=u[j], cov=sigma[j])
        
        # Calculate the responsibility using the cluster likelihood and prior probability
        responsibilities[i, j] = pi[j] * bernoulli_likelihood * mvn_likelihood

    # Normalize the responsibilities for each observation
    responsibilities[i, :] /= sum(responsibilities[i, :])

# M-Step
# Update cluster priors
pi_new = responsibilities.mean(axis=0)

# Update Bernoulli parameters
p_new = (responsibilities[:, 0] @ observations[:, 0]) / responsibilities[:, 0].sum(), (responsibilities[:, 1] @ observations[:, 0]) / responsibilities[:, 1].sum()

# Update Gaussian parameters
u_new = np.array([responsibilities[:, 0] @ observations[:, 1:] / responsibilities[:, 0].sum(), responsibilities[:, 1] @ observations[:, 1:] / responsibilities[:, 1].sum()])
sigma_new = np.array([(((responsibilities[:, 0] * (observations[:, 1:] - u_new[0]).T) @ (observations[:, 1:] - u_new[0])) / responsibilities[:, 0].sum()), (((responsibilities[:, 1] * (observations[:, 1:] - u_new[1]).T) @ (observations[:, 1:] - u_new[1])) / responsibilities[:, 1].sum())])

# Print the updated parameters
print("Updated Pi:", pi_new)
print("Updated P:", p_new)
print("Updated U:", u_new)
print("Updated Sigma:", sigma_new)
print()
print("------------------------------------------------------------------------------------------------------------------")

# E-Step and M-Step for Cluster 2
responsibilities = np.zeros((len(observations), 2))

# E-Step for Cluster 2
for i in range(len(observations)):
    for j in range(2):
        # Calculate the likelihood of the observation under the j-th cluster
        bernoulli_likelihood = p[j] ** observations[i, 0] * (1 - p[j]) ** (1 - observations[i, 0])
        mvn_likelihood = multivariate_normal.pdf(observations[i, 1:], mean=u[j], cov=sigma[j])
        
        # Calculate the responsibility using the cluster likelihood and prior probability
        responsibilities[i, j] = pi[j] * bernoulli_likelihood * mvn_likelihood

    # Normalize the responsibilities for each observation
    responsibilities[i, :] /= sum(responsibilities[i, :])

# Update cluster priors
pi_new_2 = responsibilities.mean(axis=0)

# Update Bernoulli parameters
p_new_2 = (responsibilities[:, 0] @ observations[:, 0]) / responsibilities[:, 0].sum(), (responsibilities[:, 1] @ observations[:, 0]) / responsibilities[:, 1].sum()

# Update Gaussian parameters
u_new_2 = np.array([responsibilities[:, 0] @ observations[:, 1:] / responsibilities[:, 0].sum(), responsibilities[:, 1] @ observations[:, 1:] / responsibilities[:, 1].sum()])
sigma_new_2 = np.array([(((responsibilities[:, 0] * (observations[:, 1:] - u_new_2[0]).T) @ (observations[:, 1:] - u_new_2[0])) / responsibilities[:, 0].sum()), (((responsibilities[:, 1] * (observations[:, 1:] - u_new_2[1]).T) @ (observations[:, 1:] - u_new_2[1])) / responsibilities[:, 1].sum())])

# Print the updated parameters for Cluster 2
print("Updated U for Cluster 2:", u_new_2)
print("Updated Sigma for Cluster 2:", sigma_new_2)



""" Certainly, I'll explain the code line by line:

1. Import the required libraries:
   - `numpy` for numerical operations.
   - `multivariate_normal` from `scipy.stats` to work with multivariate normal distributions.

3-6. Define the given parameters:
   - `pi`: The initial cluster priors (mixture coefficients) for two clusters.
   - `p`: The initial Bernoulli distribution parameters for both clusters.
   - `u`: The initial means for the multivariate Gaussian distributions for both clusters.
   - `sigma`: The initial covariance matrices for the multivariate Gaussian distributions for both clusters.

8-11. Define the observed data:
   - `observations`: A 2D array containing the observed data points. Each row represents an observation with the first element as a binary value (0 or 1) and the next two elements as real numbers.

13. Initialize a variable `responsibilities` as a 2D array filled with zeros. It will store the responsibilities of each observation for each cluster.

15-32. E-Step (Expectation Step):
   - This loop iterates through each observation and cluster.
   - `bernoulli_likelihood` calculates the likelihood of the binary observation under the Bernoulli distribution of the j-th cluster.
   - `mvn_likelihood` calculates the likelihood of the real-valued part of the observation under the multivariate Gaussian distribution of the j-th cluster.
   - `responsibilities` are calculated by multiplying the prior probability (π), Bernoulli likelihood, and Gaussian likelihood for each observation and each cluster.
   - The responsibilities are normalized for each observation to ensure they sum to 1.

34-36. M-Step (Maximization Step) - Update cluster priors (π):
   - `pi_new` calculates the new cluster priors by taking the mean of responsibilities for each cluster across all observations.

38-41. M-Step - Update Bernoulli parameters (p):
   - `p_new` calculates the new Bernoulli parameters for each cluster by taking the weighted average of the binary observations.

43-46. M-Step - Update Gaussian parameters (u and sigma):
   - `u_new` calculates the new means for the multivariate Gaussian distributions by taking the weighted average of the real-valued parts of the observations.
   - `sigma_new` calculates the new covariance matrices for the multivariate Gaussian distributions based on the weighted sum of squared differences between the observations and the new means.

49-52. Print the updated parameters for Cluster 1:
   - Print the updated priors, Bernoulli parameters, means, and covariance matrices.

54-55. Print a separator line to distinguish between Cluster 1 and Cluster 2 updates.

57-60. Reinitialize the `responsibilities` variable to zeros for Cluster 2.

62-92. Repeat the E-Step and M-Step for Cluster 2:
   - Calculate responsibilities, update priors, Bernoulli parameters, means, and covariance matrices for Cluster 2.

94-97. Print the updated parameters for Cluster 2:
   - Print the updated means and covariance matrices for Cluster 2.

The code performs one iteration of the EM algorithm for both clusters, updating the parameters based on the observed data and responsibilities. """

print("------------------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------------") 
# New observation
x_new = np.array([1, 0.3, 0.7])

# Initialize an array to store the posteriors
posteriors = np.zeros(2)

for j in range(2):
    # Calculate the likelihood of the new observation under the j-th cluster
    bernoulli_likelihood = p_new[j] ** x_new[0] * (1 - p_new[j]) ** (1 - x_new[0])
    print("bernoulli_likelihood", bernoulli_likelihood)
    mvn_likelihood = multivariate_normal.pdf(x_new[1:], mean=u_new[j], cov=sigma_new[j])
    print("mvn_likelihood", mvn_likelihood)
    
    # Calculate the posterior probability using the cluster likelihood and prior probability
    posteriors[j] = pi_new[j] * bernoulli_likelihood * mvn_likelihood
    print("posteriors", posteriors[j])

# Normalize the posteriors to make them sum to 1
posteriors /= posteriors.sum()

# Print the posteriors
print("Cluster Memberships (Posteriors) for the New Observation:")
print("Cluster 1:", posteriors[0])
print("Cluster 2:", posteriors[1])

print("------------------------------------------------------------------------------------------------------------------")


