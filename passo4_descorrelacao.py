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

# Vetores centrados na média
z1 = x - np.mean(x)
z2 = y - np.mean(y)
Z = np.vstack([z1, z2])

# Matriz de covariância original
K_Z = np.cov(Z)

# Diagonalização formal do JJW (Cap. 6): K_Z = Q * Lambda * Q^T
autovalores, Q = np.linalg.eigh(K_Z)

# Transformação ortogonal: W = Q^T * Z
W = Q.T @ Z
w1, w2 = W[0], W[1]
K_W = np.cov(W)

# Correlação cruzada normalizada ANTES e DEPOIS da descorrelação
r_antes = np.correlate(z1, z2, mode="full") / np.sqrt(np.sum(z1**2) * np.sum(z2**2))
r_depois = np.correlate(w1, w2, mode="full") / np.sqrt(np.sum(w1**2) * np.sum(w2**2))

lags = np.arange(-N + 1, N)
centro = N - 1

# Grava a memória de cálculo formal
np.set_printoptions(suppress=True, precision=4)
texto = (
    "=====================================================\n"
    "PASSO 4: DESCORRELAÇÃO 2-A-2 (MATRIZ DE COVARIÂNCIA)\n"
    "=====================================================\n\n"
    "1. Matriz de covariância original (K_Z):\n"
    f"{K_Z}\n\n"
    f"Coeficiente de correlação original no lag 0 (rho_XY): {r_antes[centro]:.6f}\n\n"
    "2. Autovalores (variâncias das componentes descorrelacionadas):\n"
    f"lambda_1 = {autovalores[0]:.4f}\n"
    f"lambda_2 = {autovalores[1]:.4f}\n\n"
    "3. Matriz modal Q (autovetores em colunas):\n"
    f"{Q}\n\n"
    "4. Matriz de covariância transformada (K_W = Q^T * K_Z * Q):\n"
    f"{K_W}\n\n"
    f"Coeficiente de correlação resultante no lag 0: {r_depois[centro]:.10e} (exatamente 0)\n\n"
    "Conclusão: O termo fora da diagonal de K_W é rigorosamente nulo,\n"
    "e a correlação cruzada no lag zero bate exatamente em 0.0000,\n"
    "comprovando a descorrelação completa das componentes W1 e W2.\n"
)

print(texto)

with open(pasta_passo4 / "descorrelacao.txt", "w", encoding="utf-8") as f:
    f.write(texto)

# 1. Gráfico da Correlação Cruzada Antes vs Depois - ZOOM (-250 a +250)
max_lag = 250
lags_zoom = np.arange(-max_lag, max_lag + 1)
antes_zoom = r_antes[centro - max_lag : centro + max_lag + 1]
depois_zoom = r_depois[centro - max_lag : centro + max_lag + 1]

fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=120)

axs[0].plot(lags_zoom, antes_zoom, color="purple", linewidth=1.2)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].plot(0, r_antes[centro], "ro", label=f"Lag 0: {r_antes[centro]:.4f}")
axs[0].set_title("Correlação Cruzada ANTES da Descorrelação (Zoom)")
axs[0].set_ylabel("Coeficiente")
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend(loc="upper right")

axs[1].plot(lags_zoom, depois_zoom, color="green", linewidth=1.2)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].plot(0, r_depois[centro], "go", label=f"Lag 0: {r_depois[centro]:.4f} (Zerado)")
axs[1].set_title("Correlação Cruzada DEPOIS da Descorrelação (Zoom)")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("Coeficiente")
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(pasta_passo4 / "descorrelacao_cruzada_zoom.png", dpi=150)
plt.close()

# 2. Gráfico da Correlação Cruzada Antes vs Depois - Varredura Completa
fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=120)

axs[0].plot(lags, r_antes, color="purple", linewidth=0.8)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].set_title("Correlação Cruzada ANTES da Descorrelação (Varredura Completa)")
axs[0].set_ylabel("Coeficiente")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags, r_depois, color="green", linewidth=0.8)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].set_title("Correlação Cruzada DEPOIS da Descorrelação (Varredura Completa)")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("Coeficiente")
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo4 / "descorrelacao_cruzada_completa.png", dpi=150)
plt.close()

# 3. Gráfico de dispersão opcional (nuvem de pontos)
passo = 5
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
axs[0].scatter(Z[0, ::passo], Z[1, ::passo], s=4, alpha=0.3, color="red")
axs[0].set_title("Dispersão Antes (Z1, Z2)")
axs[0].set_xlabel("Z1")
axs[0].set_ylabel("Z2")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].scatter(W[0, ::passo], W[1, ::passo], s=4, alpha=0.3, color="blue")
axs[1].set_title("Dispersão Depois (W1, W2)")
axs[1].set_xlabel("W1")
axs[1].set_ylabel("W2")
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo4 / "descorrelacao_dispersao.png", dpi=150)
plt.close()

print("Passo 4 concluído!")
print(f"Resultados salvos em: {pasta_passo4}")
