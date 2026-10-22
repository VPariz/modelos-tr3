import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. CARREGAR OS SINAIS
# ==========================================
x = np.loadtxt('startrek_h263_64k.txt')    # H.263 (~26k)
y_original = np.loadtxt('startrek_mpeg4_hq.txt') # MPEG-4 (~90k)

N = len(x)
N_y = len(y_original)

# Reamostragem para a correlação cruzada (igualar tamanhos)
y_resampled = signal.resample(y_original, N)

# Centrar na média
x_c = x - np.mean(x)
y_c_original = y_original - np.mean(y_original)
y_c_resampled = y_resampled - np.mean(y_resampled)

# ==========================================
# 2. CALCULAR AS CORRELAÇÕES (VARREDURA COMPLETA)
# ==========================================
# Autocorrelação H.263 (tamanho ~52k)
r_xx = np.correlate(x_c, x_c, mode='full')
# Autocorrelação MPEG-4 (tamanho ~180k)
r_yy = np.correlate(y_c_original, y_c_original, mode='full')
# Correlação Cruzada (tamanho ~52k)
r_xy = np.correlate(x_c, y_c_resampled, mode='full')

# ==========================================
# 3. NORMALIZAÇÃO
# ==========================================
pico_x = r_xx[N - 1]
pico_y = r_yy[N_y - 1]
pico_y_resampled = np.sum(y_c_resampled**2)

fator_xx = r_xx / pico_x
fator_yy = r_yy / pico_y
fator_xy = r_xy / np.sqrt(pico_x * pico_y_resampled)

# Criar os vetores de Lags
lags_x = np.arange(-N + 1, N)
lags_y = np.arange(-N_y + 1, N_y)

# ==========================================
# 4. GERAR O PDF COM AS DUAS VERSÕES
# ==========================================
with PdfPages('FACs_StarTrek_Completo.pdf') as pdf:

    # ---------------------------------------------------------
    # PÁGINA 1: ZOOM (-250 A 250)
    # ---------------------------------------------------------
    max_lag = 250
    centro_x = N - 1
    centro_y = N_y - 1

    xx_zoom = fator_xx[centro_x - max_lag : centro_x + max_lag + 1]
    yy_zoom = fator_yy[centro_y - max_lag : centro_y + max_lag + 1]
    xy_zoom = fator_xy[centro_x - max_lag : centro_x + max_lag + 1]
    lags_zoom = np.arange(-max_lag, max_lag + 1)

    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    fig1.suptitle('FACs - Zoom (-250 a 250 amostras)', fontsize=16, fontweight='bold')

    axs1[0].plot(lags_zoom, xx_zoom, color='red')
    axs1[0].set_title('Autocorrelação H.263 (Zoom)')
    axs1[0].grid(True)

    axs1[1].plot(lags_zoom, yy_zoom, color='blue')
    axs1[1].set_title('Autocorrelação MPEG-4 (Zoom)')
    axs1[1].grid(True)

    axs1[2].plot(lags_zoom, xy_zoom, color='purple')
    axs1[2].set_title('Correlação Cruzada (Zoom)')
    axs1[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axs1[2].grid(True)

    pdf.savefig(fig1)
    plt.close(fig1)

    # ---------------------------------------------------------
    # PÁGINA 2: VARREDURA COMPLETA (TODAS AS AMOSTRAS)
    # ---------------------------------------------------------
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    fig2.suptitle('FACs - Varredura Completa de Amostras', fontsize=16, fontweight='bold')

    axs2[0].plot(lags_x, fator_xx, color='red')
    axs2[0].set_title('Autocorrelação H.263 (Completa ~26k)')
    axs2[0].grid(True)

    axs2[1].plot(lags_y, fator_yy, color='blue')
    axs2[1].set_title('Autocorrelação MPEG-4 (Completa ~90k)')
    axs2[1].grid(True)

    axs2[2].plot(lags_x, fator_xy, color='purple')
    axs2[2].set_title('Correlação Cruzada (Completa ~26k)')
    axs2[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axs2[2].grid(True)

    pdf.savefig(fig2)
    plt.close(fig2)

print("Arquivo 'FACs_StarTrek_Completo.pdf' gerado com sucesso!")