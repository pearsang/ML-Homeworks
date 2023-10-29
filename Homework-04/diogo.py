import numpy as np

from scipy.stats import bernoulli, multivariate_normal
from sklearn.metrics import pairwise_distances

# Corre o código com o comando: python3 Homework\ 4/pen\&paper.py > Homework\ 4/resultados.txt 

#DADOS

x0 = (1, 0.6, 0.1)                      #x0[0] = y1, x0[1] = y2, x0[2] = y3
x1 = (0, -0.4, 0.8)
x2 = (0, 0.2, 0.5)
x3 = (1, 0.4, -0.1)

dados = [x0, x1, x2, x3]

u1 = np.array([1, 1])
u2 = np.array([0, 0])

cov1 = np.matrix([[2, 0.5], [0.5, 2]])
cov2 = np.matrix([[1.5, 1], [1, 1.5]])

#DISTRIBUIÇÕES

ber1 = bernoulli(0.3)
ber2 = bernoulli(0.7)

mvn1 = multivariate_normal(u1, cov1)
mvn2 = multivariate_normal(u2, cov2)

pi_1 = {"prior": 0.5, "binomial": ber1, "normal": mvn1}
pi_2 = {"prior": 0.5, "binomial": ber2, "normal": mvn2}

#FUNÇÕES

def calcula_posterior(x):

    prob_1 = pi_1["prior"] * pi_1["binomial"].pmf(x[0]) * pi_1["normal"].pdf(x[1:3])
    prob_2 = pi_2["prior"] * pi_2["binomial"].pmf(x[0]) * pi_2["normal"].pdf(x[1:3])

    pi_1_x = prob_1/(prob_1 + prob_2)
    pi_2_x = prob_2/(prob_1 + prob_2)

    return {"pi_1": pi_1_x, "pi_2": pi_2_x}

def update_priors():
    return {"p1_1": N_k["pi_1"]/N, "p1_2": N_k["pi_2"]/N}

def update_mean():

    def aux(cluster, variavel):
        num = 0

        for index, posterior in enumerate(posteriores):
            num += posterior[cluster] * dados[index][variavel]

        return num/N_k[cluster]

    pi_1_mean = {"p_sucesso": aux("pi_1", 0), "mean": [aux("pi_1", 1), aux("pi_1", 2)]}
    pi_2_mean = {"p_sucesso": aux("pi_2", 0), "mean": [aux("pi_2", 1), aux("pi_2", 2)]}

    return {"pi_1": pi_1_mean, "pi_2": pi_2_mean}

def update_variance():
    def aux(cluster):
        num = 0

        for index, posterior in enumerate(posteriores):
            num += posterior[cluster] * np.outer(dados[index][1:3], dados[index][1:3])

        return num/N_k[cluster]

    return {"pi_1": aux("pi_1"), "pi_2": aux("pi_2")}

def calculate_silhouette(a, b):
    if (a < b):
        return 1 - a/b
    else:
        return b/a - 1

# ALÍNEA A

#E-STEP

posteriores = []
for x in dados:
    posteriores.append(calcula_posterior(x))

print("EXERCÍCIO 1\n")
print("\tPosteriores:")
for index, posterior in enumerate(posteriores):
    print(f"\tx{index}: \t cluster 1 = {round(posterior['pi_1'], 2)} \t cluster 2 = {round(posterior['pi_2'], 2)}")
print()

#M-STEP

N_k = {
    "pi_1": sum(posterior["pi_1"] for posterior in posteriores),
    "pi_2": sum(posterior["pi_2"] for posterior in posteriores)
}
N = N_k["pi_1"] + N_k["pi_2"]

updated_priors = update_priors()
updated_mean = update_mean()
updated_variance = update_variance()

print(f"\tUpdated priors: \t cluster 1 = {round(updated_priors['p1_1'], 2)} \t cluster 2 = {round(updated_priors['p1_2'], 2)}\n")
print(f"\tUpdated p_sucesso: \t cluster 1 = {round(updated_mean['pi_1']['p_sucesso'], 2)} \t cluster 2 = {round(updated_mean['pi_2']['p_sucesso'], 2)}\n")
print(f"\tUpdated mean: \t\t cluster 1 = {updated_mean['pi_1']['mean']} \t cluster 2 = {updated_mean['pi_2']['mean']}\n")
print(f"\tUpdated variance: \t cluster 1 = {updated_variance['pi_1'][0]} \t\t\t\t\t\t cluster 2 = {updated_variance['pi_2'][0]}")
print(f"\t\t\t\t\t\t\t\t     {updated_variance['pi_1'][1]}\t\t\t\t\t\t\t\t     {updated_variance['pi_2'][1]}\n")


