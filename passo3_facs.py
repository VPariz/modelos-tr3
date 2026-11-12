import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from pathlib import Path

# Definicao de pastas
pasta_base = Path(__file__).resolve().parent
pasta_dados = pasta_base / "dados"
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_passo3 = pasta_base / "resultados" / "passo3_facs"
pasta_passo3_extras = pasta_passo3 / "extras"
pasta_passo3.mkdir(parents=True, exist_ok=True)
pasta_passo3_extras.mkdir(parents=True, exist_ok=True)

# Leitura dos sinais: menor e H.263, maior e MPEG-4
sinais = [np.loadtxt(arq) for arq in arquivos]
nomes = [arq.stem for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
x, y = sinais[ordem[0]], sinais[ordem[1]]
nome_x, nome_y = nomes[ordem[0]], nomes[ordem[1]]

N = len(x)      # ~26.192 (H.263)
N_y = len(y)    # ~89.998 (MPEG-4)

# Momentos globais de ambos os sinais
m_x, v_x, s_x, vmq_x = np.mean(x), np.var(x), np.std(x), np.mean(x**2)
m_y, v_y, s_y, vmq_y = np.mean(y), np.var(y), np.std(y), np.mean(y**2)

# =========================================================================
# SELECAO DO BLOCO CONTINUO DE MPEG-4 CASANDO MEDIA E DESVIO
# =========================================================================
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

# Refinamento fino
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
m_b, v_b, s_b, vmq_b, rms_b = np.mean(y_bloco), np.var(y_bloco), np.std(y_bloco), np.mean(y_bloco**2), np.sqrt(np.mean(y_bloco**2))

relatorio_bloco = (
    f"Bloco selecionado do MPEG-4: k = {melhor_k} ate {melhor_k + N} (N = {N})\n\n"
    f"{'Parametro':<15} | {'MPEG-4 Completo':<18} | {'Bloco (k=' + str(melhor_k) + ')':<18} | {'Erro (%)':<10}\n"
    f"{'-'*15}-|-{'-'*18}-|-{'-'*18}-|-{'-'*10}\n"
    f"{'Media':<15} | {m_y:<18.4f} | {m_b:<18.4f} | {abs(m_b - m_y)/m_y*100:>8.2f} %\n"
    f"{'Variancia':<15} | {v_y:<18.4f} | {v_b:<18.4f} | {abs(v_b - v_y)/v_y*100:>8.2f} %\n"
    f"{'Desvio Padrao':<15} | {s_y:<18.4f} | {s_b:<18.4f} | {abs(s_b - s_y)/s_y*100:>8.2f} %\n"
    f"{'VMQ':<15} | {vmq_y:<18.4f} | {vmq_b:<18.4f} | {abs(vmq_b - vmq_y)/vmq_y*100:>8.2f} %\n"
    f"{'RMS':<15} | {np.sqrt(vmq_y):<18.4f} | {rms_b:<18.4f} | {abs(rms_b - np.sqrt(vmq_y))/np.sqrt(vmq_y)*100:>8.2f} %\n"
)
print(relatorio_bloco)

with open(pasta_passo3 / "comparacao_valores_esperados_bloco.txt", "w", encoding="utf-8") as f:
    f.write(relatorio_bloco)

# =========================================================================
# 1. AUTOCORRELACAO NORMALIZADA MANTENDO MEDIA E VARIANCIA (COM NIVEL DC)
# Teoria (Leon-Garcia, Papoulis, Profa. Rosangela):
# R_X[k] = E[X[n]X[n+k]], com R_X[0] = E[X^2] = VMQ = sigma^2 + mu^2
# Normalizada por VMQ: r_X[k] = R_X[k] / R_X[0]
#   - Lag 0: r_X[0] = 1.0 (potencia total normalizada)
#   - Lag infinito: r_X[inf] = mu^2 / VMQ (potencia DC)
#   - Variacao AC: sigma^2 / VMQ
# =========================================================================
k_x = np.arange(N)
k_y = np.arange(N_y)

# Estimador de autocovariancia/autocorrelacao biased (padrao xcorr)
rx_biased = signal.correlate(x, x, mode="full", method="fft")[N - 1 :] / N
fac_x_dc = rx_biased / rx_biased[0]

ry_biased = signal.correlate(y, y, mode="full", method="fft")[N_y - 1 :] / N_y
fac_y_dc = ry_biased / ry_biased[0]

# Correlacao cruzada com nivel DC mantido
ry_bloco_biased0 = np.mean(y_bloco**2)
rxy_biased = signal.correlate(x, y_bloco, mode="full", method="fft")[N - 1 :] / N
fac_xy_dc = rxy_biased / np.sqrt(rx_biased[0] * ry_bloco_biased0)

# =========================================================================
# 2. VERSAO CENTRADA NA MEDIA (AUTOCOVARIANCIA / COEFICIENTE DE CORRELACAO)
# rho_X[k] = K_X[k] / sigma^2, com rho_X[0] = 1.0 e decaimento para zero
# =========================================================================
x_c = x - m_x
y_c = y - m_y
y_bloco_c = y_bloco - m_b

rxc_biased = signal.correlate(x_c, x_c, mode="full", method="fft")[N - 1 :] / N
fac_x_ac = rxc_biased / rxc_biased[0]

ryc_biased = signal.correlate(y_c, y_c, mode="full", method="fft")[N_y - 1 :] / N_y
fac_y_ac = ryc_biased / ryc_biased[0]

rxyc_biased = signal.correlate(x_c, y_bloco_c, mode="full", method="fft")[N - 1 :] / N
fac_xy_ac = rxyc_biased / np.sqrt(rxc_biased[0] * np.var(y_bloco))

# Zooms positivos (0 a 800, alinhado ao Grupo 3)
max_zoom = 800
lags_zoom = np.arange(0, max_zoom + 1)

# Suavizacao de controle de taxa de periodo 3 no H.263
idx3_x = np.arange(0, N, 3)
fac_x_dc_smooth = np.interp(np.arange(N), idx3_x, fac_x_dc[idx3_x])
fac_x_dc_smooth[0] = 1.0

# Normalizacao central pelo ponto medio dos ciclos GoP no MPEG-4 ('cara la no meio')
lags_mid_zoom = [0]
vals_mid_zoom = [1.0]
for b in range(1, (max_zoom + 1) // 12):
    k0 = b * 12
    k1 = min(k0 + 12, max_zoom + 1)
    sub = fac_y_dc[k0:k1]
    lags_mid_zoom.append(k0 + 6)
    vals_mid_zoom.append((np.min(sub) + np.max(sub)) / 2.0)
midline_zoom = np.interp(lags_zoom, lags_mid_zoom, vals_mid_zoom)

# Varredura completa central
lags_mid_comp = [0]
vals_mid_comp = [1.0]
step_gop = 12
for b in range(1, N_y // step_gop):
    k0 = b * step_gop
    k1 = min(k0 + step_gop, N_y)
    sub = fac_y_dc[k0:k1]
    lags_mid_comp.append(k0 + step_gop // 2)
    vals_mid_comp.append((np.min(sub) + np.max(sub)) / 2.0)
midline_comp = np.interp(k_y, lags_mid_comp, vals_mid_comp)

dc_nivel_x = m_x**2 / vmq_x
dc_nivel_y = m_y**2 / vmq_y

def plotar_fac(lags, valores, titulo, ylabel, cor, arquivo_saida, linha_dc=None):
    plt.figure(figsize=(11, 5), dpi=120)
    plt.plot(lags, valores, color=cor, linewidth=1.1)
    if linha_dc is not None:
        plt.axhline(linha_dc, color="black", linestyle="--", linewidth=0.8,
                    label=f"Nivel DC: $\\mu^2 / \\mathrm{{VMQ}} = {linha_dc:.4f}$")
        plt.legend(loc="upper right")
    else:
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(titulo)
    plt.xlabel("Lag (k >= 0)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=150)
    plt.close()

# ---------------------------------------------------------
# GRAFICOS PRINCIPAIS: NORMALIZADOS COM NIVEL DC (PRESERVANDO MEDIA E VARIANCIA)
# ---------------------------------------------------------
# 1. Varredura Completa
plotar_fac(k_x, fac_x_dc, f"Autocorrelacao Completa: {nome_x} (~26k)", "r_{XX}(k)", "red",
           pasta_passo3 / f"fac_{nome_x}_completa.png", linha_dc=dc_nivel_x)
plotar_fac(k_y, fac_y_dc, f"Autocorrelacao Completa: {nome_y} (~90k)", "r_{YY}(k)", "blue",
           pasta_passo3 / f"fac_{nome_y}_completa.png", linha_dc=dc_nivel_y)
plotar_fac(k_x, fac_xy_dc, f"Correlacao Cruzada Completa: {nome_x} x Bloco {nome_y} (~26k)", "r_{XY}(k)", "purple",
           pasta_passo3 / "fac_cruzada_completa.png")

# 2. Zooms Positivos (0 a 800)
# H.263 com decaimento suave (suavizacao de artefato PB de 3 quadros)
plt.figure(figsize=(8, 4.2), dpi=150)
plt.plot(lags_zoom, fac_x_dc_smooth[:max_zoom + 1], color="red", linewidth=1.1, label=f"{nome_x} (FAC normalizada)")
plt.axhline(dc_nivel_x, color="darkred", linestyle="--", linewidth=0.8, label=f"Nivel DC: {dc_nivel_x:.4f}")
plt.title(f"Autocorrelacao {nome_x} — Visao Aproximada (0 a {max_zoom})")
plt.xlabel("Deslocamento de tempo (Lag k)")
plt.ylabel("Autocorrelacao Normalizada")
plt.xlim(0, max_zoom)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(pasta_passo3 / f"fac_{nome_x}_zoom.png", dpi=150)
plt.close()

# MPEG-4 com oscilacao de GoP N=12 e linha media central ('cara la no meio')
plt.figure(figsize=(8, 4.2), dpi=150)
plt.plot(lags_zoom, fac_y_dc[:max_zoom + 1], color="#60a5fa", linewidth=0.7, alpha=0.75, label="FAC MPEG-4 (oscilacao GoP, N = 12)")
plt.plot(lags_zoom, midline_zoom, color="#1e3a8a", linewidth=1.8, label="Linha media central (ponto medio dos ciclos)")
plt.axhline(dc_nivel_y, color="darkblue", linestyle="--", linewidth=0.8, label=f"Nivel DC: {dc_nivel_y:.4f}")
plt.title(f"Autocorrelacao {nome_y} — Visao Aproximada (0 a {max_zoom})")
plt.xlabel("Deslocamento de tempo (Lag k)")
plt.ylabel("Autocorrelacao Normalizada")
plt.xlim(0, max_zoom)
plt.ylim(0.60, 1.02)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(pasta_passo3 / f"fac_{nome_y}_zoom.png", dpi=150)
plt.close()

# MPEG-4 Completa com linha media central
plt.figure(figsize=(8, 4.2), dpi=150)
plt.plot(k_y, fac_y_dc, color="#60a5fa", linewidth=0.4, alpha=0.6, label="FAC MPEG-4 completa (~90k)")
plt.plot(k_y, midline_comp, color="#1e3a8a", linewidth=1.5, label="Linha media central")
plt.axhline(dc_nivel_y, color="darkblue", linestyle="--", linewidth=0.8, label=f"Nivel DC: {dc_nivel_y:.4f}")
plt.title(f"Autocorrelacao {nome_y} — Varredura Completa (~90k)")
plt.xlabel("Deslocamento de tempo (Lag k)")
plt.ylabel("Autocorrelacao Normalizada")
plt.xlim(0, N_y)
plt.ylim(0.55, 1.02)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(pasta_passo3 / f"fac_{nome_y}_completa.png", dpi=150)
plt.close()

# Cruzada zoom (0 a 250)
plotar_fac(lags_zoom[:251], fac_xy_dc[:251], f"Correlacao Cruzada Zoom (0 a 250): {nome_x} x Bloco {nome_y}",
           "r_{XY}(k)", "purple", pasta_passo3 / "fac_cruzada_zoom.png")

# ---------------------------------------------------------
# GRAFICOS 3 EM 1 (COM NIVEL DC) - PRINCIPAL
# ---------------------------------------------------------
# 3 em 1 Zoom
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_zoom, fac_x_dc[:max_zoom + 1], color="red", linewidth=1.2)
axs[0].axhline(dc_nivel_x, color="darkred", linestyle="--", linewidth=0.8, label=f"Nivel DC ({dc_nivel_x:.4f})")
axs[0].set_title(f"Autocorrelacao Zoom: {nome_x} (Nivel DC Mantido)")
axs[0].set_ylabel("r_{XX}(k)")
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend(loc="upper right")

axs[1].plot(lags_zoom, fac_y_dc[:max_zoom + 1], color="blue", linewidth=1.2)
axs[1].axhline(dc_nivel_y, color="darkblue", linestyle="--", linewidth=0.8, label=f"Nivel DC ({dc_nivel_y:.4f})")
axs[1].set_title(f"Autocorrelacao Zoom: {nome_y} (Preservando Media e Variancia - Picos do GoP N=12)")
axs[1].set_ylabel("r_{YY}(k)")
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend(loc="upper right")

axs[2].plot(lags_zoom, fac_xy_dc[:max_zoom + 1], color="purple", linewidth=1.2)
axs[2].set_title(f"Correlacao Cruzada Zoom: {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].set_ylabel("r_{XY}(k)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_zoom.png", dpi=150)
plt.close()

# 3 em 1 Completo
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(k_x, fac_x_dc, color="red", linewidth=0.8)
axs[0].axhline(dc_nivel_x, color="darkred", linestyle="--", linewidth=0.8, label=f"Nivel DC ({dc_nivel_x:.4f})")
axs[0].set_title(f"Autocorrelacao Completa: {nome_x} (~26k)")
axs[0].set_ylabel("r_{XX}(k)")
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend(loc="upper right")

axs[1].plot(k_y, fac_y_dc, color="blue", linewidth=0.8)
axs[1].axhline(dc_nivel_y, color="darkblue", linestyle="--", linewidth=0.8, label=f"Nivel DC ({dc_nivel_y:.4f})")
axs[1].set_title(f"Autocorrelacao Completa: {nome_y} (~90k)")
axs[1].set_ylabel("r_{YY}(k)")
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend(loc="upper right")

axs[2].plot(k_x, fac_xy_dc, color="purple", linewidth=0.8)
axs[2].set_title(f"Correlacao Cruzada Completa: {nome_x} x Bloco {nome_y} (~26k)")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].set_ylabel("r_{XY}(k)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_completa.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAFICOS EXTRAS: VERSAO SEM NIVEL DC (AUTOCOVARIANCIA PURA AC)
# ---------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=120)
axs[0].plot(lags_zoom, fac_x_ac[:max_zoom + 1], color="red", linewidth=1.2)
axs[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[0].set_title(f"Autocovariancia Zoom (Sem DC): {nome_x}")
axs[0].set_ylabel("\\rho_{XX}(k)")
axs[0].grid(True, linestyle="--", alpha=0.6)

axs[1].plot(lags_zoom, fac_y_ac[:max_zoom + 1], color="blue", linewidth=1.2)
axs[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[1].set_title(f"Autocovariancia Zoom (Sem DC): {nome_y}")
axs[1].set_ylabel("\\rho_{YY}(k)")
axs[1].grid(True, linestyle="--", alpha=0.6)

axs[2].plot(lags_zoom, fac_xy_ac[:max_zoom + 1], color="purple", linewidth=1.2)
axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axs[2].set_title(f"Covariancia Cruzada Zoom (Sem DC): {nome_x} x Bloco {nome_y}")
axs[2].set_xlabel("Lag (k >= 0)")
axs[2].set_ylabel("\\rho_{XY}(k)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(pasta_passo3_extras / "facs_3_em_1_zoom_sem_dc.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# PDF COMPILADO COM AS DUAS ABORDAGENS
# ---------------------------------------------------------
with PdfPages(pasta_passo3 / "FACs_StarTrek_Completo.pdf") as pdf:
    # Pagina 1: Zoom com nivel DC (Canonica Profa. Rosangela)
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 11), dpi=120)
    fig1.suptitle("FACs - Zoom (0 a 250) [Nivel DC Mantido - Preservando Media e Variancia]", fontsize=13, fontweight="bold")

    axs1[0].plot(lags_zoom, fac_x_dc[:max_zoom + 1], color="red", linewidth=1.1)
    axs1[0].axhline(dc_nivel_x, color="darkred", linestyle="--", linewidth=0.8, label=f"DC: {dc_nivel_x:.4f}")
    axs1[0].set_title(f"Autocorrelacao Zoom: {nome_x}")
    axs1[0].set_ylabel("r_{XX}(k)")
    axs1[0].grid(True, linestyle="--", alpha=0.6)
    axs1[0].legend(loc="upper right")

    axs1[1].plot(lags_zoom, fac_y_dc[:max_zoom + 1], color="blue", linewidth=1.1)
    axs1[1].axhline(dc_nivel_y, color="darkblue", linestyle="--", linewidth=0.8, label=f"DC: {dc_nivel_y:.4f}")
    axs1[1].set_title(f"Autocorrelacao Zoom: {nome_y} (Picos do GoP N=12)")
    axs1[1].set_ylabel("r_{YY}(k)")
    axs1[1].grid(True, linestyle="--", alpha=0.6)
    axs1[1].legend(loc="upper right")

    axs1[2].plot(lags_zoom, fac_xy_dc[:max_zoom + 1], color="purple", linewidth=1.1)
    axs1[2].set_title(f"Correlacao Cruzada Zoom: {nome_x} x Bloco {nome_y}")
    axs1[2].set_xlabel("Lag (k >= 0)")
    axs1[2].set_ylabel("r_{XY}(k)")
    axs1[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    # Pagina 2: Varredura Completa com Nivel DC
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 11), dpi=120)
    fig2.suptitle("FACs - Varredura Completa [Nivel DC Mantido]", fontsize=13, fontweight="bold")

    axs2[0].plot(k_x, fac_x_dc, color="red", linewidth=0.8)
    axs2[0].axhline(dc_nivel_x, color="darkred", linestyle="--", linewidth=0.8, label=f"DC: {dc_nivel_x:.4f}")
    axs2[0].set_title(f"Autocorrelacao Completa: {nome_x} (~26k)")
    axs2[0].set_ylabel("r_{XX}(k)")
    axs2[0].grid(True, linestyle="--", alpha=0.6)
    axs2[0].legend(loc="upper right")

    axs2[1].plot(k_y, fac_y_dc, color="blue", linewidth=0.8)
    axs2[1].axhline(dc_nivel_y, color="darkblue", linestyle="--", linewidth=0.8, label=f"DC: {dc_nivel_y:.4f}")
    axs2[1].set_title(f"Autocorrelacao Completa: {nome_y} (~90k)")
    axs2[1].set_ylabel("r_{YY}(k)")
    axs2[1].grid(True, linestyle="--", alpha=0.6)
    axs2[1].legend(loc="upper right")

    axs2[2].plot(k_x, fac_xy_dc, color="purple", linewidth=0.8)
    axs2[2].set_title(f"Correlacao Cruzada Completa: {nome_x} x Bloco {nome_y} (~26k)")
    axs2[2].set_xlabel("Lag (k >= 0)")
    axs2[2].set_ylabel("r_{XY}(k)")
    axs2[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

    # Pagina 3: Zoom Sem Nivel DC (Autocovariancia / AC pura)
    fig3, axs3 = plt.subplots(3, 1, figsize=(10, 11), dpi=120)
    fig3.suptitle("FACs - Zoom (0 a 250) [Sem Nivel DC - Flutuacoes AC Puras]", fontsize=13, fontweight="bold")

    axs3[0].plot(lags_zoom, fac_x_ac[:max_zoom + 1], color="red", linewidth=1.1)
    axs3[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs3[0].set_title(f"Autocovariancia: {nome_x}")
    axs3[0].set_ylabel("\\rho_{XX}(k)")
    axs3[0].grid(True, linestyle="--", alpha=0.6)

    axs3[1].plot(lags_zoom, fac_y_ac[:max_zoom + 1], color="blue", linewidth=1.1)
    axs3[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs3[1].set_title(f"Autocovariancia: {nome_y}")
    axs3[1].set_ylabel("\\rho_{YY}(k)")
    axs3[1].grid(True, linestyle="--", alpha=0.6)

    axs3[2].plot(lags_zoom, fac_xy_ac[:max_zoom + 1], color="purple", linewidth=1.1)
    axs3[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs3[2].set_title(f"Covariancia Cruzada: {nome_x} x Bloco {nome_y}")
    axs3[2].set_xlabel("Lag (k >= 0)")
    axs3[2].set_ylabel("\\rho_{XY}(k)")
    axs3[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close(fig3)

print("Passo 3 concluido com sucesso!")
