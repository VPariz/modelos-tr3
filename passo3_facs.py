from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

pasta_base = Path(__file__).resolve().parent
pasta_dados = pasta_base / "dados"
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_passo3 = pasta_base / "resultados" / "passo3_facs"
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

relatorio_bloco = (
    f"Bloco selecionado do MPEG-4: k = {melhor_k} ate {melhor_k + N} (N = {N})\n\n"
    f"{'Parametro':<15} | {'MPEG-4 Completo':<18} | {'Bloco (k=' + str(melhor_k) + ')':<18} | {'Erro (%)':<10}\n"
    f"{'-'*15}-|-{'-'*18}-|-{'-'*18}-|-{'-'*10}\n"
    f"{'Media':<15} | {m_y:<18.4f} | {m_b:<18.4f} | {abs(m_b - m_y)/m_y*100:>8.2f} %\n"
    f"{'Variancia':<15} | {v_y:<18.4f} | {v_b:<18.4f} | {abs(v_b - v_y)/v_y*100:>8.2f} %\n"
    f"{'Desvio Padrao':<15} | {s_y:<18.4f} | {s_b:<18.4f} | {abs(s_b - s_y)/s_y*100:>8.2f} %\n"
    f"{'VMQ':<15} | {vmq_y:<18.4f} | {vmq_b:<18.4f} | {abs(vmq_b - vmq_y)/vmq_y*100:>8.2f} %\n"
    f"{'RMS':<15} | {rms_y:<18.4f} | {rms_b:<18.4f} | {abs(rms_b - rms_y)/rms_y*100:>8.2f} %\n"
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
# AUTOCORRELACAO NORMALIZADA TIRANDO O NIVEL DC (MEDIA SUBTRAIDA)
# O nivel DC e removido para que a curva meca a correlacao das flutuacoes AC,
# com pico = 1.0 no lag 0 e decaindo em direcao a zero.
# =========================================================================
x_c = x - np.mean(x)
y_c = y - np.mean(y)
y_bloco_c = y_bloco - np.mean(y_bloco)

k_x = np.arange(N)
k_y = np.arange(N_y)

# Estimador de autocovariancia normalizado pelo pico em k=0
r_xx = np.correlate(x_c, x_c, mode="full")[N - 1 :] / (N - k_x)
fac_xx = r_xx / r_xx[0]

r_yy = np.correlate(y_c, y_c, mode="full")[N_y - 1 :] / (N_y - k_y)
fac_yy = r_yy / r_yy[0]

r_xy = np.correlate(x_c, y_bloco_c, mode="full")[N - 1 :] / (N - k_x)
fac_xy = r_xy / np.sqrt(r_xx[0] * (np.var(y_bloco)))

# zooms positivos (0 a 250)
max_zoom = 250
lags_zoom = np.arange(0, max_zoom + 1)
xx_zoom = fac_xx[: max_zoom + 1]
yy_zoom = fac_yy[: max_zoom + 1]
xy_zoom = fac_xy[: max_zoom + 1]

def plotar_fac_sem_dc(lags, valores, titulo, cor, arquivo_saida):
    plt.figure(figsize=(11, 5), dpi=120)
    plt.plot(lags, valores, color=cor, linewidth=1.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(titulo)
    plt.xlabel("Lag (k >= 0)")
    plt.ylabel("Coeficiente")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=150)
    plt.close()

# 1. Graficos Principais (normalizados sem nivel DC - varredura completa)
plotar_fac_sem_dc(k_x, fac_xx, f"Autocorrelacao: {nome_x}", "red", pasta_passo3 / f"fac_{nome_x}_completa.png")
plotar_fac_sem_dc(k_y, fac_yy, f"Autocorrelacao: {nome_y}", "blue", pasta_passo3 / f"fac_{nome_y}_completa.png")
plotar_fac_sem_dc(k_x, fac_xy, f"Correlacao Cruzada: {nome_x} x Bloco {nome_y}", "purple", pasta_passo3 / "fac_cruzada_completa.png")

# 2. Graficos Principais (normalizados sem nivel DC - zoom 0 a 250)
plotar_fac_sem_dc(lags_zoom, xx_zoom, f"Autocorrelacao Zoom (0 a 250): {nome_x}", "red", pasta_passo3 / f"fac_{nome_x}_zoom.png")
plotar_fac_sem_dc(lags_zoom, yy_zoom, f"Autocorrelacao Zoom (0 a 250): {nome_y}", "blue", pasta_passo3 / f"fac_{nome_y}_zoom.png")
plotar_fac_sem_dc(lags_zoom, xy_zoom, f"Correlacao Cruzada Zoom (0 a 250): {nome_x} x Bloco {nome_y}", "purple", pasta_passo3 / "fac_cruzada_zoom.png")

# 3 em 1 completo (salvo em extras)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(k_x, fac_xx, color="red", linewidth=0.8)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].set_title(f"Autocorrelacao: {nome_x}")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(k_y, fac_yy, color="blue", linewidth=0.8)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].set_title(f"Autocorrelacao: {nome_y}")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(k_x, fac_xy, color="purple", linewidth=0.8)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title(f"Correlacao Cruzada: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_completa.png", dpi=150)
plt.close()

# 3 em 1 zoom (salvo em extras)
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_zoom, xx_zoom, color="red", linewidth=1.2)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].set_title(f"Autocorrelacao Zoom: {nome_x}")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_zoom, yy_zoom, color="blue", linewidth=1.2)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].set_title(f"Autocorrelacao Zoom: {nome_y}")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(lags_zoom, xy_zoom, color="purple", linewidth=1.2)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title(f"Correlacao Cruzada Zoom: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_zoom.png", dpi=150)
plt.close()

# exporta PDF com zoom e varredura completa reunidos
with PdfPages(pasta_passo3 / "FACs_StarTrek_Completo.pdf") as pdf:
    # Pagina 1: Zoom (0 a 250)
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 11), dpi=120)
    fig1.suptitle("FACs - Zoom (0 a 250 amostras) [Nivel DC Removido]", fontsize=14, fontweight="bold")
    axs1[0].plot(lags_zoom, xx_zoom, color="red", linewidth=1.1)
    axs1[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs1[0].set_title(f"Autocorrelacao Zoom: {nome_x}")
    axs1[0].set_ylabel("Coeficiente")
    axs1[0].grid(True, linestyle="--", alpha=0.6)

    axs1[1].plot(lags_zoom, yy_zoom, color="blue", linewidth=1.1)
    axs1[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs1[1].set_title(f"Autocorrelacao Zoom: {nome_y}")
    axs1[1].set_ylabel("Coeficiente")
    axs1[1].grid(True, linestyle="--", alpha=0.6)

    axs1[2].plot(lags_zoom, xy_zoom, color="purple", linewidth=1.1)
    axs1[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs1[2].set_title(f"Correlacao Cruzada Zoom: {nome_x} x Bloco {nome_y}")
    axs1[2].set_xlabel("Lag (k >= 0)")
    axs1[2].set_ylabel("Coeficiente")
    axs1[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    # Pagina 2: Varredura Completa
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 11), dpi=120)
    fig2.suptitle("FACs - Varredura Completa [Nivel DC Removido]", fontsize=14, fontweight="bold")
    axs2[0].plot(k_x, fac_xx, color="red", linewidth=0.8)
    axs2[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs2[0].set_title(f"Autocorrelacao Completa: {nome_x} (~26k)")
    axs2[0].set_ylabel("Coeficiente")
    axs2[0].grid(True, linestyle="--", alpha=0.6)

    axs2[1].plot(k_y, fac_yy, color="blue", linewidth=0.8)
    axs2[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs2[1].set_title(f"Autocorrelacao Completa: {nome_y} (~90k)")
    axs2[1].set_ylabel("Coeficiente")
    axs2[1].grid(True, linestyle="--", alpha=0.6)

    axs2[2].plot(k_x, fac_xy, color="purple", linewidth=0.8)
    axs2[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs2[2].set_title(f"Correlacao Cruzada Completa: {nome_x} x Bloco {nome_y} (~26k)")
    axs2[2].set_xlabel("Lag (k >= 0)")
    axs2[2].set_ylabel("Coeficiente")
    axs2[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

print("Passo 3 concluido.")
