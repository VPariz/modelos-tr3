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

N = len(x)      # ~26k (H.263)
N_y = len(y)    # ~90k (MPEG-4)

# =========================================================================
# 1. SELEÇÃO DO BLOCO CORTADO DO MPEG-4 COM MESMO TAMANHO E MESMA PDF
# =========================================================================
# Busca o bloco contínuo de tamanho N que melhor preserva média e desvio
m_y = np.mean(y)
s_y = np.std(y)

melhor_k = 0
menor_erro = float("inf")
for k_cand in range(0, N_y - N + 1, 50):
    cand = y[k_cand : k_cand + N]
    erro = np.sqrt(((np.mean(cand) - m_y) / s_y)**2 + ((np.std(cand) - s_y) / s_y)**2)
    if erro < menor_erro:
        menor_erro = erro
        melhor_k = k_cand

# Bloco cortado com N amostras consecutivas
y_bloco = y[melhor_k : melhor_k + N]

print(f"Bloco selecionado do MPEG-4: inicio no indice k = {melhor_k}")
print(f"  MPEG-4 Original (90k):  Media = {m_y:.2f}, Desvio = {s_y:.2f}")
print(f"  MPEG-4 Bloco ({N}):   Media = {np.mean(y_bloco):.2f}, Desvio = {np.std(y_bloco):.2f}")

# Gráfico de Verificação da Manutenção da PDF (Histograma sobreposto)
plt.figure(figsize=(10, 5), dpi=120)
plt.hist(y, bins=150, density=True, color="blue", alpha=0.45, label=f"MPEG-4 Completo (~90k) [m={m_y:.1f}, s={s_y:.1f}]")
plt.hist(y_bloco, bins=150, density=True, histtype="step", color="red", linewidth=1.8, label=f"MPEG-4 Bloco Cortado (~26k, k={melhor_k}) [m={np.mean(y_bloco):.1f}, s={np.std(y_bloco):.1f}]")
plt.title("Verificacao da Manutencao da PDF: MPEG-4 Original vs Bloco Cortado")
plt.xlabel("Valor")
plt.ylabel("Densidade")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(pasta_passo3 / "verificacao_pdf_bloco_mpeg4.png", dpi=150)
plt.close()

# =========================================================================
# 2. CÁLCULO DAS 3 FACS (AUTOCORRELAÇÕES E CORRELAÇÃO CRUZADA)
# =========================================================================
# Sinais centrados na média
x_c = x - np.mean(x)
y_c = y - np.mean(y)
y_bloco_c = y_bloco - np.mean(y_bloco)

# Autocorrelações completas
r_xx = np.correlate(x_c, x_c, mode="full")
r_yy = np.correlate(y_c, y_c, mode="full")

# Correlação cruzada feita com o bloco cortado de mesmo tamanho (N)
r_xy = np.correlate(x_c, y_bloco_c, mode="full")

# Normalização pelo pico na origem (coeficiente = 1 no lag 0)
fac_xx = r_xx / r_xx[N - 1]
fac_yy = r_yy / r_yy[N_y - 1]
fac_xy = r_xy / np.sqrt(np.sum(x_c**2) * np.sum(y_bloco_c**2))

# =========================================================================
# 3. PARTE POSITIVA DOS GRÁFICOS (conforme orientação da professora)
# =========================================================================
# Como a autocorrelação é função par, a parte positiva (lags >= 0) contém toda a informação
lags_pos_x = np.arange(0, N)
lags_pos_y = np.arange(0, N_y)

fac_xx_pos = fac_xx[N - 1 :]
fac_yy_pos = fac_yy[N_y - 1 :]
fac_xy_pos = fac_xy[N - 1 :]

# Zooms positivos (lag 0 a 250)
max_zoom = 250
lags_zoom_pos = np.arange(0, max_zoom + 1)
xx_zoom_pos = fac_xx_pos[: max_zoom + 1]
yy_zoom_pos = fac_yy_pos[: max_zoom + 1]
xy_zoom_pos = fac_xy_pos[: max_zoom + 1]

def plotar_positivo(lags, valores, titulo, cor, arquivo_saida):
    plt.figure(figsize=(11, 5), dpi=120)
    plt.plot(lags, valores, color=cor, linewidth=0.9)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(titulo)
    plt.xlabel("Lag (k >= 0)")
    plt.ylabel("Coeficiente")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(pasta_passo3 / arquivo_saida, dpi=150)
    plt.close()

# Gráficos individuais positivos - Varredura completa
plotar_positivo(lags_pos_x, fac_xx_pos, f"Autocorrelacao (Parte Positiva): {nome_x}", "red", f"fac_{nome_x}_positiva_completa.png")
plotar_positivo(lags_pos_y, fac_yy_pos, f"Autocorrelacao (Parte Positiva): {nome_y}", "blue", f"fac_{nome_y}_positiva_completa.png")
plotar_positivo(lags_pos_x, fac_xy_pos, f"Correlacao Cruzada (Parte Positiva): {nome_x} x Bloco {nome_y}", "purple", "fac_cruzada_positiva_completa.png")

# Gráficos individuais positivos - Zoom (0 a 250)
plotar_positivo(lags_zoom_pos, xx_zoom_pos, f"Autocorrelacao Zoom (0 a 250): {nome_x}", "red", f"fac_{nome_x}_zoom_positivo.png")
plotar_positivo(lags_zoom_pos, yy_zoom_pos, f"Autocorrelacao Zoom (0 a 250): {nome_y}", "blue", f"fac_{nome_y}_zoom_positivo.png")
plotar_positivo(lags_zoom_pos, xy_zoom_pos, "Correlacao Cruzada Zoom (0 a 250)", "purple", "fac_cruzada_zoom_positivo.png")

# Gráfico 3-em-1 Positivo Completo (pronto para o relatório)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_pos_x, fac_xx_pos, color="red", linewidth=0.8)
axs[0].set_title(f"Autocorrelacao: {nome_x}")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_pos_y, fac_yy_pos, color="blue", linewidth=0.8)
axs[1].set_title(f"Autocorrelacao: {nome_y}")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(lags_pos_x, fac_xy_pos, color="purple", linewidth=0.8)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title(f"Correlacao Cruzada: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3 / "facs_3_em_1_positiva_completa.png", dpi=150)
plt.close()

# Gráfico 3-em-1 Positivo com Zoom (0 a 250)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_zoom_pos, xx_zoom_pos, color="red", linewidth=1.2)
axs[0].set_title(f"Autocorrelacao Zoom: {nome_x}")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_zoom_pos, yy_zoom_pos, color="blue", linewidth=1.2)
axs[1].set_title(f"Autocorrelacao Zoom: {nome_y}")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(lags_zoom_pos, xy_zoom_pos, color="purple", linewidth=1.2)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title(f"Correlacao Cruzada Zoom: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3 / "facs_3_em_1_zoom_positivo.png", dpi=150)
plt.close()

print("Passo 3 concluído!")
print(f"Graficos e verificacao da PDF salvos em: {pasta_passo3}")
