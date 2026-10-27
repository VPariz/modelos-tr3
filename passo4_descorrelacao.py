from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

pasta_base = Path(__file__).resolve().parent
pasta_dados = pasta_base / "dados"
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_passo4 = pasta_base / "resultados" / "passo4_descorrelacao"
pasta_passo4_extras = pasta_passo4 / "extras"
pasta_passo4.mkdir(parents=True, exist_ok=True)
pasta_passo4_extras.mkdir(parents=True, exist_ok=True)

# le os dois sinais
sinais = [np.loadtxt(arq) for arq in arquivos]
nomes = [arq.stem for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
x, y = sinais[ordem[0]], sinais[ordem[1]]
nome_x, nome_y = nomes[ordem[0]], nomes[ordem[1]]

N = len(x)
N_y = len(y)

# seleciona o mesmo bloco do mpeg-4 que casa a media e o desvio
m_y = np.mean(y)
s_y = np.std(y)

melhor_k = 0
menor_erro = float("inf")
for k_cand in range(0, N_y - N + 1, 10):
    cand = y[k_cand : k_cand + N]
    m_c = np.mean(cand)
    s_c = np.std(cand)
    erro = np.sqrt(((m_c - m_y) / s_y)**2 + ((s_c - s_y) / s_y)**2)
    if erro < menor_erro:
        menor_erro = erro
        melhor_k = k_cand

k_ini = max(0, melhor_k - 30)
k_fim = min(N_y - N + 1, melhor_k + 30)
for k_cand in range(k_ini, k_fim):
    cand = y[k_cand : k_cand + N]
    m_c = np.mean(cand)
    s_c = np.std(cand)
    erro = np.sqrt(((m_c - m_y) / s_y)**2 + ((s_c - s_y) / s_y)**2)
    if erro < menor_erro:
        menor_erro = erro
        melhor_k = k_cand

y_bloco = y[melhor_k : melhor_k + N]

# sinais centrados na media
z1 = x - np.mean(x)
z2 = y_bloco - np.mean(y_bloco)
Z = np.vstack([z1, z2])

# matriz de covariancia original
K_Z = np.cov(Z)

# autovalores e matriz de autovetores Q (diagonalizacao)
autovalores, Q = np.linalg.eigh(K_Z)

# transformacao para descorrelacionar: W = Q^T * Z
W = Q.T @ Z
w1, w2 = W[0], W[1]
K_W = np.cov(W)

# correlacao cruzada normalizada antes e depois
r_antes = np.correlate(z1, z2, mode="full") / np.sqrt(np.sum(z1**2) * np.sum(z2**2))
r_depois = np.correlate(w1, w2, mode="full") / np.sqrt(np.sum(w1**2) * np.sum(w2**2))

centro = N - 1

# grava memoria de calculo
np.set_printoptions(suppress=True, precision=4)
texto = (
    "PASSO 4: DESCORRELACAO 2-A-2 (MATRIZ DE COVARIANCIA)\n\n"
    f"Sinal 1: {nome_x} (N = {N})\n"
    f"Sinal 2: Bloco continuo de {nome_y} (k = {melhor_k} a {melhor_k + N})\n\n"
    "1. Matriz de covariancia original (K_Z):\n"
    f"{K_Z}\n\n"
    f"Coeficiente de correlacao original (lag 0): {r_antes[centro]:.6f}\n\n"
    "2. Autovalores (variancias das componentes):\n"
    f"lambda_1 = {autovalores[0]:.4f}\n"
    f"lambda_2 = {autovalores[1]:.4f}\n\n"
    "3. Matriz modal Q (autovetores):\n"
    f"{Q}\n\n"
    "4. Matriz de covariancia transformada (K_W = Q^T * K_Z * Q):\n"
    f"{K_W}\n\n"
    f"Coeficiente de correlacao resultante no lag 0: {r_depois[centro]:.4e} (aproximadamente 0)\n"
)

print(texto)

with open(pasta_passo4 / "descorrelacao.txt", "w", encoding="utf-8") as f:
    f.write(texto)

# zoom da correlacao antes e depois (0 a 250)
max_zoom = 250
lags_zoom_pos = np.arange(0, max_zoom + 1)
antes_zoom_pos = r_antes[centro : centro + max_zoom + 1]
depois_zoom_pos = r_depois[centro : centro + max_zoom + 1]

fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=120)

axs[0].plot(lags_zoom_pos, antes_zoom_pos, color="purple", linewidth=1.2)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].plot(0, r_antes[centro], "ro", label=f"Lag 0: {r_antes[centro]:.4f}")
axs[0].set_title("Correlacao Cruzada ANTES da Descorrelacao (Zoom)")
axs[0].set_ylabel("Coeficiente")
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend(loc="upper right")

axs[1].plot(lags_zoom_pos, depois_zoom_pos, color="green", linewidth=1.2)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].plot(0, r_depois[centro], "go", label=f"Lag 0: {r_depois[centro]:.4f}")
axs[1].set_title("Correlacao Cruzada DEPOIS da Descorrelacao (Zoom)")
axs[1].set_xlabel("Lag (k >= 0)")
axs[1].set_ylabel("Coeficiente")
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(pasta_passo4 / "descorrelacao_cruzada_zoom.png", dpi=150)
plt.close()

# varredura completa da correlacao antes e depois
lags_pos = np.arange(0, N)
antes_pos = r_antes[centro : centro + N]
depois_pos = r_depois[centro : centro + N]

fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=120)

axs[0].plot(lags_pos, antes_pos, color="purple", linewidth=0.8)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].set_title("Correlacao Cruzada ANTES da Descorrelacao (Completa)")
axs[0].set_ylabel("Coeficiente")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_pos, depois_pos, color="green", linewidth=0.8)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].set_title("Correlacao Cruzada DEPOIS da Descorrelacao (Completa)")
axs[1].set_xlabel("Lag (k >= 0)")
axs[1].set_ylabel("Coeficiente")
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo4 / "descorrelacao_cruzada_completa.png", dpi=150)
plt.close()

# zoom simetrico em extras (-250 a +250)
lags_zoom_sim = np.arange(-max_zoom, max_zoom + 1)
antes_zoom_sim = r_antes[centro - max_zoom : centro + max_zoom + 1]
depois_zoom_sim = r_depois[centro - max_zoom : centro + max_zoom + 1]

fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=120)

axs[0].plot(lags_zoom_sim, antes_zoom_sim, color="purple", linewidth=1.2)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].plot(0, r_antes[centro], "ro", label=f"Lag 0: {r_antes[centro]:.4f}")
axs[0].set_title("Correlacao Cruzada ANTES (Zoom Simetrico)")
axs[0].set_ylabel("Coeficiente")
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend(loc="upper right")

axs[1].plot(lags_zoom_sim, depois_zoom_sim, color="green", linewidth=1.2)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].plot(0, r_depois[centro], "go", label=f"Lag 0: {r_depois[centro]:.4f}")
axs[1].set_title("Correlacao Cruzada DEPOIS (Zoom Simetrico)")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("Coeficiente")
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(pasta_passo4_extras / "descorrelacao_cruzada_zoom_simetrico.png", dpi=150)
plt.close()

# nuvem de pontos em extras
passo = 5
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
axs[0].scatter(Z[0, ::passo], Z[1, ::passo], s=4, alpha=0.3, color="red")
axs[0].set_title("Dispersao Antes (Z1, Z2)")
axs[0].set_xlabel("Z1")
axs[0].set_ylabel("Z2")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].scatter(W[0, ::passo], W[1, ::passo], s=4, alpha=0.3, color="blue")
axs[1].set_title("Dispersao Depois (W1, W2)")
axs[1].set_xlabel("W1")
axs[1].set_ylabel("W2")
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo4_extras / "descorrelacao_dispersao.png", dpi=150)
plt.close()

# exporta PDF com correlacoes cruzadas e dispersao reunidas
with PdfPages(pasta_passo4 / "Descorrelacao_StarTrek_Completo.pdf") as pdf:
    # Pagina 1: Correlacao Cruzada Antes vs Depois (Zoom)
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8), dpi=120)
    fig1.suptitle("Descorrelacao 2-a-2: Correlacao Cruzada Antes vs Depois", fontsize=14, fontweight="bold")
    axs1[0].plot(lags_zoom_pos, antes_zoom_pos, color="purple", linewidth=1.2)
    axs1[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs1[0].plot(0, r_antes[centro], "ro", label=f"Lag 0: {r_antes[centro]:.4f}")
    axs1[0].set_title("Correlacao Cruzada ANTES da Descorrelacao (Zoom)")
    axs1[0].set_ylabel("Coeficiente")
    axs1[0].grid(True, linestyle="--", alpha=0.6)
    axs1[0].legend(loc="upper right")

    axs1[1].plot(lags_zoom_pos, depois_zoom_pos, color="green", linewidth=1.2)
    axs1[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs1[1].plot(0, r_depois[centro], "go", label=f"Lag 0: {r_depois[centro]:.4f}")
    axs1[1].set_title("Correlacao Cruzada DEPOIS da Descorrelacao (Zoom)")
    axs1[1].set_xlabel("Lag (k >= 0)")
    axs1[1].set_ylabel("Coeficiente")
    axs1[1].grid(True, linestyle="--", alpha=0.6)
    axs1[1].legend(loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    # Pagina 2: Nuvem de Pontos (Dispersao Z1,Z2 vs W1,W2)
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    fig2.suptitle("Dispersao no Plano: Antes (Z1, Z2) vs Depois (W1, W2)", fontsize=14, fontweight="bold")
    axs2[0].scatter(Z[0, ::passo], Z[1, ::passo], s=4, alpha=0.3, color="red")
    axs2[0].set_title("Dispersao Antes (Z1, Z2)")
    axs2[0].set_xlabel("Z1")
    axs2[0].set_ylabel("Z2")
    axs2[0].grid(True, linestyle="--", alpha=0.6)

    axs2[1].scatter(W[0, ::passo], W[1, ::passo], s=4, alpha=0.3, color="blue")
    axs2[1].set_title("Dispersao Depois (W1, W2)")
    axs2[1].set_xlabel("W1")
    axs2[1].set_ylabel("W2")
    axs2[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

print("Passo 4 concluido.")
