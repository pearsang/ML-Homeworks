import numpy as np
from scipy.stats import multivariate_normal

# Define the values of y1 and y2
y1 = [0.54, 0.66, 0.76, 0.41]
y2 = [0.11, 0.39, 0.28, 0.53]

# Combine y1 and y2 into a single array for multivariate analysis
y1_y2 = np.column_stack((y1, y2))

# Calculate the mean and covariance matrix for y1 and y2
mean_y1_y2 = np.mean(y1_y2, axis=0)
cov_y1_y2 = np.cov(y1_y2, rowvar=False)
print(mean_y1_y2)
print(cov_y1_y2)

mean_vector = np.array([0.5925, 0.3275])
covariance_matrix = np.array([[ 0.02289167, -0.00975833], [-0.00975833, 0.03149167]])

# Create a multivariate normal distribution
mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)

# Test values
test_values = np.array([0.42, 0.59])

# Calculate the probability density function (PDF) for the test values
pdf_value = mvn.pdf(test_values)

print(f"The PDF value for the test values {test_values} is: {pdf_value}")