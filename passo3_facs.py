from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

pasta_dados = Path("dados")
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_passo3 = Path("resultados/passo3_facs")
pasta_passo3.mkdir(parents=True, exist_ok=True)

# Lê os dois sinais (o menor vira x, o maior vira y)
sinais = [np.loadtxt(arq) for arq in arquivos]
nomes = [arq.stem for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
x, y = sinais[ordem[0]], sinais[ordem[1]]
nome_x, nome_y = nomes[ordem[0]], nomes[ordem[1]]

N = len(x)
N_y = len(y)

# Pega N amostras reais do y distribuídas uniformemente
indices = np.linspace(0, N_y - 1, N).astype(int)
y_ajustado = y[indices]

# Tira a média
x_c = x - np.mean(x)
y_c = y - np.mean(y)
y_aj_c = y_ajustado - np.mean(y_ajustado)

# Correlações
r_xx = np.correlate(x_c, x_c, mode="full")
r_yy = np.correlate(y_c, y_c, mode="full")
r_xy = np.correlate(x_c, y_aj_c, mode="full")

# Normaliza pelo valor na origem (coeficiente entre -1 e 1)
fac_xx = r_xx / r_xx[N - 1]
fac_yy = r_yy / r_yy[N_y - 1]
fac_xy = r_xy / np.sqrt(np.sum(x_c**2) * np.sum(y_aj_c**2))

lags_x = np.arange(-N + 1, N)
lags_y = np.arange(-N_y + 1, N_y)

# Zoom de -250 a +250
max_lag = 250
z = np.arange(-max_lag, max_lag + 1)
xx_zoom = fac_xx[N - 1 - max_lag : N + max_lag]
yy_zoom = fac_yy[N_y - 1 - max_lag : N_y + max_lag]
xy_zoom = fac_xy[N - 1 - max_lag : N + max_lag]


def plotar(lags, valores, titulo, cor, arquivo_saida):
    plt.figure(figsize=(12, 5), dpi=120)
    plt.plot(lags, valores, color=cor, linewidth=0.8)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(titulo)
    plt.xlabel("Lag")
    plt.ylabel("Coeficiente")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(pasta_passo3 / arquivo_saida, dpi=150)
    plt.close()


# Gráficos individuais completos
plotar(lags_x, fac_xx, f"Autocorrelação Completa: {nome_x}", "red", f"fac_{nome_x}_completa.png")
plotar(lags_y, fac_yy, f"Autocorrelação Completa: {nome_y}", "blue", f"fac_{nome_y}_completa.png")
plotar(lags_x, fac_xy, f"Correlação Cruzada Completa: {nome_x} x {nome_y}", "purple", "fac_cruzada_completa.png")

# Gráficos individuais com zoom
plotar(z, xx_zoom, f"Autocorrelação Zoom (-250 a 250): {nome_x}", "red", f"fac_{nome_x}_zoom.png")
plotar(z, yy_zoom, f"Autocorrelação Zoom (-250 a 250): {nome_y}", "blue", f"fac_{nome_y}_zoom.png")
plotar(z, xy_zoom, "Correlação Cruzada Zoom (-250 a 250)", "purple", "fac_cruzada_zoom.png")

# Gráfico 3-em-1 Completo (pronto para o relatório)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_x, fac_xx, color="red", linewidth=0.8)
axs[0].set_title(f"Autocorrelação: {nome_x}")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_y, fac_yy, color="blue", linewidth=0.8)
axs[1].set_title(f"Autocorrelação: {nome_y}")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(lags_x, fac_xy, color="purple", linewidth=0.8)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title("Correlação Cruzada")
axs[2].set_xlabel("Lag")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3 / "facs_3_em_1_completa.png", dpi=150)
plt.close()

# Gráfico 3-em-1 com Zoom (-250 a 250) (pronto para o relatório)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(z, xx_zoom, color="red", linewidth=1.2)
axs[0].set_title(f"Autocorrelação Zoom: {nome_x}")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(z, yy_zoom, color="blue", linewidth=1.2)
axs[1].set_title(f"Autocorrelação Zoom: {nome_y}")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(z, xy_zoom, color="purple", linewidth=1.2)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title("Correlação Cruzada Zoom")
axs[2].set_xlabel("Lag")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3 / "facs_3_em_1_zoom.png", dpi=150)
plt.close()

print("Passo 3 concluído!")
print(f"Gráficos salvos em: {pasta_passo3}")
