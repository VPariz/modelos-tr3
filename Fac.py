import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. CARREGAR OS SINAIS
# ==========================================
x = np.loadtxt('startrek_h263_64k.txt')  # Sinal H.263
y = np.loadtxt('startrek_mpeg4_hq.txt')  # Sinal MPEG-4

# ==========================================
# 2. ADEQUAR O TAMANHO E REMOVER A MÉDIA
# ==========================================
N = len(x)
y_igualado = signal.resample(y, N)

x_c = x - np.mean(x)
y_c = y_igualado - np.mean(y_igualado)

# ==========================================
# 3. CALCULAR AS CORRELAÇÕES
# ==========================================
r_xx = np.correlate(x_c, x_c, mode='full')
r_yy = np.correlate(y_c, y_c, mode='full')
r_xy = np.correlate(x_c, y_c, mode='full')

# ==========================================
# 4. NORMALIZAR (FATOR DE CORRELAÇÃO)
# ==========================================
pico_x = r_xx[N - 1]
pico_y = r_yy[N - 1]

r_xx_norm = r_xx / pico_x
r_yy_norm = r_yy / pico_y
r_xy_norm = r_xy / np.sqrt(pico_x * pico_y)

# ==========================================
# 5. PREPARAR O EIXO X (LAGS DE -250 A 250)
# ==========================================
max_lag = 250
centro = N - 1
lags = np.arange(-max_lag, max_lag + 1)

r_xx_plot = r_xx_norm[centro - max_lag : centro + max_lag + 1]
r_yy_plot = r_yy_norm[centro - max_lag : centro + max_lag + 1]
r_xy_plot = r_xy_norm[centro - max_lag : centro + max_lag + 1]

# ==========================================
# 6. PLOTAR OS GRÁFICOS
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

# --- TÍTULO GERAL DA FIGURA ---
fig.suptitle('Análise de Correlação: Vídeo Star Trek', fontsize=16, fontweight='bold')

# Gráfico 1: H.263
axs[0].plot(lags, r_xx_plot, color='red')
axs[0].set_title('FAC - Autocorrelação do H.263')
axs[0].set_xlabel('Atraso (k)')
axs[0].set_ylabel('Correlação Normalizada')
axs[0].set_ylim(-0.05, 1.05)
axs[0].grid(True)

# Gráfico 2: MPEG-4
axs[1].plot(lags, r_yy_plot, color='blue')
axs[1].set_title('FAC - Autocorrelação do MPEG-4')
axs[1].set_xlabel('Atraso (k)')
axs[1].set_ylabel('Correlação Normalizada')
axs[1].set_ylim(-0.05, 1.05)
axs[1].grid(True)

# Gráfico 3: Cruzada (Escala Automática para ver as ondinhas)
axs[2].plot(lags, r_xy_plot, color='purple')
axs[2].set_title('Correlação Cruzada (H.263 vs MPEG-4)')
axs[2].set_xlabel('Atraso (k)')
axs[2].set_ylabel('Correlação Normalizada')

# Linha de referência no zero
axs[2].axhline(0, color='black', linewidth=1, linestyle='--')
axs[2].grid(True)

# Salva os arquivos
plt.savefig('FACs_StarTrek_Final.pdf', format='pdf', bbox_inches='tight')
plt.savefig('FACs_StarTrek_Final.png', format='png', dpi=300, bbox_inches='tight')

plt.show()