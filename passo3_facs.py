from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

pasta_dados = Path("dados")
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_passo3 = Path("resultados/passo3_facs")
pasta_passo3_extras = pasta_passo3 / "extras"
pasta_passo3.mkdir(parents=True, exist_ok=True)
pasta_passo3_extras.mkdir(parents=True, exist_ok=True)

# le os dois arquivos (o menor vira x, o maior vira y)
sinais = [np.loadtxt(arq) for arq in arquivos]
nomes = [arq.stem for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
x, y = sinais[ordem[0]], sinais[ordem[1]]
nome_x, nome_y = nomes[ordem[0]], nomes[ordem[1]]

N = len(x)      # ~26k (H.263)
N_y = len(y)    # ~90k (MPEG-4)

# valores esperados do sinal completo de 90k
m_y = np.mean(y)
v_y = np.var(y)
s_y = np.std(y)
vmq_y = np.mean(y**2)
rms_y = np.sqrt(vmq_y)

# varredura para achar o bloco de tamanho N com media e desvio mais proximos do global
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

# refinamento passo 1 ao redor do melhor_k
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

m_b = np.mean(y_bloco)
v_b = np.var(y_bloco)
s_b = np.std(y_bloco)
vmq_b = np.mean(y_bloco**2)
rms_b = np.sqrt(vmq_b)

# niveis DC teoricos normalizados pelo VMQ: mu^2 / E[X^2]
dc_x = (np.mean(x)**2) / np.mean(x**2)
dc_y = (m_y**2) / vmq_y
dc_b = (m_b**2) / vmq_b
dc_xy = (np.mean(x) * m_b) / np.sqrt(np.mean(x**2) * vmq_b)

relatorio_bloco = (
    f"Bloco selecionado do MPEG-4: k = {melhor_k} ate {melhor_k + N} (N = {N})\n\n"
    f"{'Parametro':<15} | {'MPEG-4 Completo':<18} | {'Bloco (k=' + str(melhor_k) + ')':<18} | {'Erro (%)':<10}\n"
    f"{'-'*15}-|-{'-'*18}-|-{'-'*18}-|-{'-'*10}\n"
    f"{'Media':<15} | {m_y:<18.4f} | {m_b:<18.4f} | {abs(m_b - m_y)/m_y*100:>8.2f} %\n"
    f"{'Variancia':<15} | {v_y:<18.4f} | {v_b:<18.4f} | {abs(v_b - v_y)/v_y*100:>8.2f} %\n"
    f"{'Desvio Padrao':<15} | {s_y:<18.4f} | {s_b:<18.4f} | {abs(s_b - s_y)/s_y*100:>8.2f} %\n"
    f"{'VMQ':<15} | {vmq_y:<18.4f} | {vmq_b:<18.4f} | {abs(vmq_b - vmq_y)/vmq_y*100:>8.2f} %\n"
    f"{'RMS':<15} | {rms_y:<18.4f} | {rms_b:<18.4f} | {abs(rms_b - rms_y)/rms_y*100:>8.2f} %\n\n"
    f"Niveis DC Teoricos (mu^2 / VMQ):\n"
    f"  - H.263:  {dc_x:.4f} ({dc_x*100:.1f}% da potencia total)\n"
    f"  - MPEG-4: {dc_y:.4f} ({dc_y*100:.1f}% da potencia total)\n"
    f"  - Cruzada: {dc_xy:.4f}\n"
)

print(relatorio_bloco)

with open(pasta_passo3 / "comparacao_valores_esperados_bloco.txt", "w", encoding="utf-8") as f:
    f.write(relatorio_bloco)

# histograma sobreposto em extras
plt.figure(figsize=(10, 5), dpi=120)
plt.hist(y, bins=150, density=True, color="blue", alpha=0.45, label=f"MPEG-4 Completo [m={m_y:.1f}, s={s_y:.1f}]")
plt.hist(y_bloco, bins=150, density=True, histtype="step", color="red", linewidth=1.8, label=f"MPEG-4 Bloco Cortado [m={m_b:.1f}, s={s_b:.1f}]")
plt.title("Verificacao da PDF: MPEG-4 Original vs Bloco Cortado")
plt.xlabel("Valor")
plt.ylabel("Densidade")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(pasta_passo3_extras / "verificacao_pdf_bloco_mpeg4.png", dpi=150)
plt.close()

# =========================================================================
# AUTOCORRELAÇÃO REAL DO SINAL (COM NÍVEL DC - R_XX[k] = E[X(n)X(n+k)])
# Como o sinal tem media nao nula, a autocorrelacao e sustentada pelo sinal DC.
# Normalizando por R(0) = VMQ, a curva fica estavel no patamar mu^2 / VMQ.
# =========================================================================

k_x = np.arange(N)
r_xx = np.correlate(x, x, mode="full")[N - 1 :] / (N - k_x)
r_xx_norm = r_xx / r_xx[0]

k_y = np.arange(N_y)
r_yy = np.correlate(y, y, mode="full")[N_y - 1 :] / (N_y - k_y)
r_yy_norm = r_yy / r_yy[0]

r_xy = np.correlate(x, y_bloco, mode="full")[N - 1 :] / (N - k_x)
r_xy_norm = r_xy / np.sqrt(r_xx[0] * np.mean(y_bloco**2))

# zooms positivos (0 a 250)
max_zoom = 250
lags_zoom = np.arange(0, max_zoom + 1)
xx_zoom = r_xx_norm[: max_zoom + 1]
yy_zoom = r_yy_norm[: max_zoom + 1]
xy_zoom = r_xy_norm[: max_zoom + 1]

def plotar_fac_dc(lags, valores, dc_ref, titulo, cor, arquivo_saida):
    plt.figure(figsize=(11, 5), dpi=120)
    plt.plot(lags, valores, color=cor, linewidth=1.0, label="FAC Normalizada")
    plt.axhline(dc_ref, color="black", linestyle="--", linewidth=1.1, label=f"Nivel DC ({dc_ref:.2f})")
    plt.title(titulo)
    plt.xlabel("Lag (k >= 0)")
    plt.ylabel("Coeficiente Normalizado")
    plt.ylim(0, 1.08)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=150)
    plt.close()

# 1. Graficos Principais com nivel DC (completos)
plotar_fac_dc(k_x, r_xx_norm, dc_x, f"Autocorrelacao (com nivel DC): {nome_x}", "red", pasta_passo3 / f"fac_{nome_x}_completa.png")
plotar_fac_dc(k_y, r_yy_norm, dc_y, f"Autocorrelacao (com nivel DC): {nome_y}", "blue", pasta_passo3 / f"fac_{nome_y}_completa.png")
plotar_fac_dc(k_x, r_xy_norm, dc_xy, f"Correlacao Cruzada (com nivel DC): {nome_x} x Bloco {nome_y}", "purple", pasta_passo3 / "fac_cruzada_completa.png")

# 2. Graficos Principais com nivel DC (zoom 0 a 250)
plotar_fac_dc(lags_zoom, xx_zoom, dc_x, f"Autocorrelacao Zoom (0 a 250): {nome_x}", "red", pasta_passo3 / f"fac_{nome_x}_zoom.png")
plotar_fac_dc(lags_zoom, yy_zoom, dc_y, f"Autocorrelacao Zoom (0 a 250): {nome_y}", "blue", pasta_passo3 / f"fac_{nome_y}_zoom.png")
plotar_fac_dc(lags_zoom, xy_zoom, dc_xy, f"Correlacao Cruzada Zoom (0 a 250): {nome_x} x Bloco {nome_y}", "purple", pasta_passo3 / "fac_cruzada_zoom.png")

# =========================================================================
# EXTRAS: Curvas centradas na media (Autocovariancia / AC sem DC)
# Salvas em extras/ caso queira ver o decaimento das flutuacoes AC
# =========================================================================
x_c = x - np.mean(x)
y_c = y - np.mean(y)
y_bloco_c = y_bloco - np.mean(y_bloco)

r_xx_ac = np.correlate(x_c, x_c, mode="full")[N - 1 :] / (N - k_x)
r_xx_ac /= r_xx_ac[0]

r_yy_ac = np.correlate(y_c, y_c, mode="full")[N_y - 1 :] / (N_y - k_y)
r_yy_ac /= r_yy_ac[0]

r_xy_ac = np.correlate(x_c, y_bloco_c, mode="full")[N - 1 :] / (N - k_x)
r_xy_ac /= np.sqrt(r_xx_ac[0] * np.var(y_bloco))

# 3 em 1 com nivel DC (salvo em extras)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(k_x, r_xx_norm, color="red", linewidth=0.8)
axs[0].axhline(dc_x, color="black", linestyle="--", linewidth=0.8, label=f"DC={dc_x:.2f}")
axs[0].set_title(f"Autocorrelacao: {nome_x}")
axs[0].set_ylim(0, 1.08)
axs[0].legend(loc="upper right")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(k_y, r_yy_norm, color="blue", linewidth=0.8)
axs[1].axhline(dc_y, color="black", linestyle="--", linewidth=0.8, label=f"DC={dc_y:.2f}")
axs[1].set_title(f"Autocorrelacao: {nome_y}")
axs[1].set_ylim(0, 1.08)
axs[1].legend(loc="upper right")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(k_x, r_xy_norm, color="purple", linewidth=0.8)
axs[2].axhline(dc_xy, color="black", linestyle="--", linewidth=0.8, label=f"DC={dc_xy:.2f}")
axs[2].set_title(f"Correlacao Cruzada: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].set_ylim(0, 1.08)
axs[2].legend(loc="upper right")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_completa.png", dpi=150)
plt.close()

# 3 em 1 zoom (salvo em extras)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_zoom, xx_zoom, color="red", linewidth=1.2)
axs[0].axhline(dc_x, color="black", linestyle="--", linewidth=0.8, label=f"DC={dc_x:.2f}")
axs[0].set_title(f"Autocorrelacao Zoom: {nome_x}")
axs[0].set_ylim(0, 1.08)
axs[0].legend(loc="upper right")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_zoom, yy_zoom, color="blue", linewidth=1.2)
axs[1].axhline(dc_y, color="black", linestyle="--", linewidth=0.8, label=f"DC={dc_y:.2f}")
axs[1].set_title(f"Autocorrelacao Zoom: {nome_y}")
axs[1].set_ylim(0, 1.08)
axs[1].legend(loc="upper right")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(lags_zoom, xy_zoom, color="purple", linewidth=1.2)
axs[2].axhline(dc_xy, color="black", linestyle="--", linewidth=0.8, label=f"DC={dc_xy:.2f}")
axs[2].set_title(f"Correlacao Cruzada Zoom: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].set_ylim(0, 1.08)
axs[2].legend(loc="upper right")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_zoom.png", dpi=150)
plt.close()

print("Passo 3 concluido.")
