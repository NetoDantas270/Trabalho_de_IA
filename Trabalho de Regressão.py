import numpy as np
import matplotlib.pyplot as plt

dados = np.loadtxt("atividade_enzimatica.csv", delimiter=',')

X = dados[:, :-1]  # Temperatura e pH
y = dados[:, -1].reshape(-1, 1)  # Atividade enzimática

X_intercepto = np.hstack([np.ones((X.shape[0], 1)), X])

# Figure 1 - Dois gráficos de espalhamento separados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: Temperatura vs. Atividade Enzimática
ax1.scatter(X[:, 0], y, color='blue', label="Dados experimentais")
ax1.set_xlabel("Temperatura (°C)")
ax1.set_ylabel("Atividade Enzimática")
ax1.set_title("Temperatura vs. Atividade Enzimática")
ax1.legend()
ax1.grid(True)

# Gráfico 2: pH vs. Atividade Enzimática
ax2.scatter(X[:, 1], y, color='red', label="Dados experimentais")
ax2.set_xlabel("pH")
ax2.set_ylabel("Atividade Enzimática")
ax2.set_title("pH vs. Atividade Enzimática")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Dimensão da matriz X (Regressoras): {X.shape}")  # N x p
print(f"Dimensão do vetor y (Variável dependente): {y.shape}")  # N x 1

# 3 questão
# Modelo 1: MQO Tradicional (OLS)
beta_ols = np.linalg.inv(X_intercepto.T @ X_intercepto) @ X_intercepto.T @ y
print(f"Coeficientes MQO Tradicional: {beta_ols.flatten()}")

# Modelo 2: MQO Regularizado (Tikhonov)
alpha1 = 1.0
I = np.eye(X_intercepto.shape[1])
I[0, 0] = 0
beta_ridge = np.linalg.inv(X_intercepto.T @ X_intercepto + alpha1 * I) @ X_intercepto.T @ y
print(f"Coeficientes MQO Regularizado (Tikhonov) com λ = 1: {beta_ridge.flatten()}")

# Modelo 3: Média dos valores observáveis
y_medio = np.mean(y)
print(f"Média dos valores observáveis: {y_medio}")

# 4 questão
lambdas = [0, 0.25, 0.5, 0.75, 1]
print("\nMQO Regularizado para diferentes valores de λ:")
for alpha2 in lambdas:
    I = np.eye(X_intercepto.shape[1])
    I[0, 0] = 0
    beta_ridge_lambda = np.linalg.inv(X_intercepto.T @ X_intercepto + alpha2 * I) @ X_intercepto.T @ y
    print(f"λ = {alpha2}: {beta_ridge_lambda.flatten()}")

# 5 questão - Validação Monte Carlo
R = 500
proporcao_treino = 0.8

rss_ols = []
rss_ridge = {alpha: [] for alpha in lambdas}
rss_media = []

np.random.seed(42)

for _ in range(R):
    indices = np.random.permutation(len(y))
    n_treino = int(proporcao_treino * len(y))
    treino_idx, teste_idx = indices[:n_treino], indices[n_treino:]

    X_treino, X_teste = X_intercepto[treino_idx], X_intercepto[teste_idx]
    y_treino, y_teste = y[treino_idx], y[teste_idx]

    # MQO Tradicional
    beta_ols_mc = np.linalg.inv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_pred_ols = X_teste @ beta_ols_mc
    rss_ols.append(np.sum((y_teste - y_pred_ols) ** 2))

    # MQO Regularizado para cada lambda
    for alpha2 in lambdas:
        I = np.eye(X_treino.shape[1])
        I[0, 0] = 0
        beta_ridge_mc = np.linalg.inv(X_treino.T @ X_treino + alpha2 * I) @ X_treino.T @ y_treino
        y_pred_ridge = X_teste @ beta_ridge_mc
        rss_ridge[alpha2].append(np.sum((y_teste - y_pred_ridge) ** 2))

    # Média dos valores observáveis
    y_pred_media = np.full(y_teste.shape, np.mean(y_treino))
    rss_media.append(np.sum((y_teste - y_pred_media) ** 2))

# 6 questão - Estatísticas e Gráficos dos resultados
rss_stats = {
    'MQO Tradicional': {
        'mean': np.mean(rss_ols),
        'std': np.std(rss_ols),
        'max': np.max(rss_ols),
        'min': np.min(rss_ols)
    }
}

for alpha2 in lambdas:
    rss_stats[f'MQO Regularizado (λ={alpha2})'] = {
        'mean': np.mean(rss_ridge[alpha2]),
        'std': np.std(rss_ridge[alpha2]),
        'max': np.max(rss_ridge[alpha2]),
        'min': np.min(rss_ridge[alpha2])
    }

rss_stats['Média Observável'] = {
    'mean': np.mean(rss_media),
    'std': np.std(rss_media),
    'max': np.max(rss_media),
    'min': np.min(rss_media)
}

# Exibir as estatísticas em formato de tabela
print("\nEstatísticas do RSS por Modelo:")
print("-" * 80)
print(f"{'Modelo':<25} | {'Média':>10} | {'Desvio Padrão':>15} | {'Máximo':>10} | {'Mínimo':>10}")
print("-" * 80)
for model, stats in rss_stats.items():
    print(f"{model:<25} | {stats['mean']:>10.4f} | {stats['std']:>15.4f} | {stats['max']:>10.4f} | {stats['min']:>10.4f}")
print("-" * 80)

labels = list(rss_stats.keys())
mean_values = [stats['mean'] for stats in rss_stats.values()]
std_values = [stats['std'] for stats in rss_stats.values()]
max_values = [stats['max'] for stats in rss_stats.values()]
min_values = [stats['min'] for stats in rss_stats.values()]

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2
ax.bar(x - bar_width * 1.5, mean_values, bar_width, label='Média')
ax.bar(x - bar_width * 0.5, std_values, bar_width, label='Desvio Padrão')
ax.bar(x + bar_width * 0.5, max_values, bar_width, label='Valor Máximo')
ax.bar(x + bar_width * 1.5, min_values, bar_width, label='Valor Mínimo')

ax.set_xlabel('Modelos')
ax.set_ylabel('Valores do RSS')
ax.set_title('Estatísticas dos RSS para diferentes Modelos')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()