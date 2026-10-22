from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

pasta_dados = Path("dados")
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_passo4 = Path("resultados/passo4_descorrelacao")
pasta_passo4.mkdir(parents=True, exist_ok=True)

# Lê os dois sinais (o menor vira x, o maior vira y)
sinais = [np.loadtxt(arq) for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
x, y = sinais[ordem[0]], sinais[ordem[1]]

# Iguala os tamanhos com amostragem uniforme direta
N = len(x)
indices = np.linspace(0, len(y) - 1, N).astype(int)
y = y[indices]

# Vetor centrado Z = [X - mX, Y - mY]
Z = np.vstack([x - np.mean(x), y - np.mean(y)])

# Matriz de covariância original
K_Z = np.cov(Z)

# Autovalores e autovetores (diagonalização formal do JJW: K_Z = Q * Lambda * Q^T)
autovalores, Q = np.linalg.eigh(K_Z)

# Transformação ortogonal: W = Q^T * Z
W = Q.T @ Z
K_W = np.cov(W)

np.set_printoptions(suppress=True, precision=4)

texto = (
    "=====================================================\n"
    "PASSO 4: DESCORRELAÇÃO 2-A-2 (MATRIZ DE COVARIÂNCIA)\n"
    "=====================================================\n\n"
    "1. Matriz de covariância original (K_Z):\n"
    f"{K_Z}\n\n"
    f"Coeficiente de correlação original (rho_XY): "
    f"{K_Z[0, 1] / np.sqrt(K_Z[0, 0] * K_Z[1, 1]):.6f}\n\n"
    "2. Autovalores (variâncias das componentes descorrelacionadas):\n"
    f"{autovalores}\n\n"
    "3. Matriz modal Q (autovetores em colunas):\n"
    f"{Q}\n\n"
    "4. Matriz de covariância transformada (K_W = Q^T * K_Z * Q):\n"
    f"{K_W}\n\n"
    "Conclusão: O termo fora da diagonal é nulo (~0), comprovando\n"
    "que as componentes W1 e W2 estão perfeitamente descorrelacionadas.\n"
)

print(texto)

with open(pasta_passo4 / "descorrelacao.txt", "w", encoding="utf-8") as f:
    f.write(texto)

# Gráfico de dispersão antes e depois da transformação (1 a cada 5 pontos para clareza)
passo = 5
fig, axs = plt.subplots(1, 2, figsize=(13, 6), dpi=120)

axs[0].scatter(Z[0, ::passo], Z[1, ::passo], s=4, alpha=0.3, color="red")
axs[0].set_title("Antes da Descorrelação (Z1, Z2)")
axs[0].set_xlabel("Z1 (Sinal 1 centrado)")
axs[0].set_ylabel("Z2 (Sinal 2 centrado)")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].scatter(W[0, ::passo], W[1, ::passo], s=4, alpha=0.3, color="blue")
axs[1].set_title("Depois da Descorrelação (W1, W2)")
axs[1].set_xlabel(f"W1 (Variância = {autovalores[0]:.1f})")
axs[1].set_ylabel(f"W2 (Variância = {autovalores[1]:.1f})")
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo4 / "descorrelacao.png", dpi=150)
plt.close()

print("Passo 4 concluído!")
print(f"Resultados salvos em: {pasta_passo4}")
