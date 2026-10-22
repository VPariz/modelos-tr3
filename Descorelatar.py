import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. CARREGAR OS SINAIS
# ==========================================
x = np.loadtxt('startrek_h263_64k.txt')
y = np.loadtxt('startrek_mpeg4_hq.txt')
N = len(x)

# ==========================================
# 2. TRATAMENTO DE TAMANHO (PULO DE AMOSTRAS)
# ==========================================
indices = np.linspace(0, len(y) - 1, N).astype(int)
y_ajustado = y[indices]

x_c = x - np.mean(x)
y_c = y_ajustado - np.mean(y_ajustado)

# ==========================================
# 3. AS CONTAS DA DESCORRELAÇÃO
# ==========================================
var_x = np.var(x_c)
cov_xy = np.cov(x_c, y_c)[0, 1]
a = cov_xy / var_x
y_desc = y_c - (a * x_c)
cov_final = np.cov(x_c, y_desc)[0, 1]

# ==========================================
# 4. CALCULAR AS CORRELAÇÕES
# ==========================================
r_xy_antes = np.correlate(x_c, y_c, mode='full')
r_xy_depois = np.correlate(x_c, y_desc, mode='full')

norm_antes = r_xy_antes / np.sqrt(np.var(x_c) * np.var(y_c) * N * N)
norm_depois = r_xy_depois / np.sqrt(np.var(x_c) * np.var(y_desc) * N * N)

# Vetor de lags completo
lags_completo = np.arange(-N + 1, N)

# ==========================================
# 5. GERAR O PDF COM DESIGN MODERNO
# ==========================================
nome_arquivo = 'Descorrelação 2-a-2 - Explicação e Resultados.pdf'

with PdfPages(nome_arquivo) as pdf:

    # ---------------------------------------------------------
    # PÁGINA 1: TEXTO COM CAIXAS COLORIDAS
    # ---------------------------------------------------------
    fig_texto = plt.figure(figsize=(8.27, 11.69), facecolor='#F8F9FA') 
    ax = fig_texto.add_axes([0.05, 0.05, 0.9, 0.9]) 
    ax.axis('off')

    ax.text(0.5, 0.96, 'RELATÓRIO DE DESCORRELAÇÃO 2-A-2', 
            ha='center', va='center', fontsize=18, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#2C3E50', edgecolor='none'))
    ax.text(0.5, 0.91, 'Análise Estocástica de Sinais de Vídeo (Star Trek)', 
            ha='center', va='center', fontsize=12, fontstyle='italic', color='#7F8C8D')

    caixa1 = dict(boxstyle="round,pad=0.6", facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.8)
    ax.text(0.05, 0.85, '1. Adequação de Tamanho (Pulo de Amostras)', 
            fontsize=12, fontweight='bold', color='#1565C0', va='top')
    texto_1 = (
        "Como o H.263 possui 26.192 amostras e o MPEG-4 possui 89.998, aplicou-se a decimação por pulo de amostras.\n"
        "Selecionou-se amostras uniformemente distribuídas por todo o filme, garantindo que ambos os sinais\n"
        "tenham o mesmo tamanho, mas sem perder a representação temporal completa da cena."
    )
    ax.text(0.07, 0.82, texto_1, fontsize=10.5, family='sans-serif', va='top', linespacing=1.5, bbox=caixa1)

    caixa2 = dict(boxstyle="round,pad=0.6", facecolor='#FFF3E0', edgecolor='#FFB74D', alpha=0.8)
    ax.text(0.05, 0.68, '2. Fundamentação Teórica (Ortogonalização de Gram-Schmidt)', 
            fontsize=12, fontweight='bold', color='#E65100', va='top')
    texto_2 = (
        "O objetivo é remover do sinal Y toda a componente linear prevista por X. Isso zera o termo fora da diagonal\n"
        "da Matriz de Covariância. Subtrai-se de Y a sua projeção sobre X, conforme a equação:\n\n"
        "                       Y_desc = Y - (a * X)\n\n"
        "Onde o coeficiente de projeção 'a' é definido por: a = Cov(X,Y) / Var(X)"
    )
    ax.text(0.07, 0.65, texto_2, fontsize=10.5, family='sans-serif', va='top', linespacing=1.5, bbox=caixa2)

    caixa3 = dict(boxstyle="round,pad=0.6", facecolor='#E8F5E9', edgecolor='#81C784', alpha=0.8)
    ax.text(0.05, 0.48, '3. Memória de Cálculo (Resultados Obtidos)', 
            fontsize=12, fontweight='bold', color='#2E7D32', va='top')
    texto_3 = (
        f"• Variância do Sinal X (Var_x) = {var_x:.4f}\n"
        f"• Covariância Original (Cov_xy) = {cov_xy:.4f}\n"
        f"• Coeficiente 'a' = {a:.6f}\n\n"
        f"Após a aplicação da fórmula, a nova covariância residual foi de: {cov_final:.6f}.\n"
        "(Valor nulo comprova matematicamente a ortogonalização dos sinais)."
    )
    ax.text(0.07, 0.45, texto_3, fontsize=10.5, family='sans-serif', va='top', linespacing=1.5, bbox=caixa3)

    caixa4 = dict(boxstyle="round,pad=0.6", facecolor='#F3E5F5', edgecolor='#CE93D8', alpha=0.8)
    ax.text(0.05, 0.28, '4. Análise dos Resultados (Gráficos - Páginas 2 e 3)', 
            fontsize=12, fontweight='bold', color='#6A1B9A', va='top')
    texto_4 = (
        "• Página 2 (Zoom): Mostra o comportamento detalhado próximo à origem.\n"
        "• Página 3 (Varredura Completa): Mostra o decaimento ao longo de todas as amostras do sinal."
    )
    ax.text(0.07, 0.25, texto_4, fontsize=10.5, family='sans-serif', va='top', linespacing=1.5, bbox=caixa4)

    pdf.savefig(fig_texto)
    plt.close(fig_texto)

    # ---------------------------------------------------------
    # PÁGINA 2: GRÁFICOS (ZOOM)
    # ---------------------------------------------------------
    max_lag = 250
    centro = N - 1
    lags_zoom = np.arange(-max_lag, max_lag + 1)
    antes_zoom = norm_antes[centro - max_lag : centro + max_lag + 1]
    depois_zoom = norm_depois[centro - max_lag : centro + max_lag + 1]

    fig_zoom, axs_zoom = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    fig_zoom.patch.set_facecolor('#F8F9FA')
    fig_zoom.suptitle('Resultados Práticos - Zoom (-250 a 250)', fontsize=16, fontweight='bold', color='#2C3E50')

    axs_zoom[0].plot(lags_zoom, antes_zoom, color='#8E44AD', linewidth=1.5)
    axs_zoom[0].set_title('Correlação Cruzada ANTES (Zoom)')
    axs_zoom[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axs_zoom[0].grid(True, linestyle=':')

    axs_zoom[1].plot(lags_zoom, depois_zoom, color='#27AE60', linewidth=1.5)
    axs_zoom[1].set_title('Correlação Cruzada DEPOIS (Zoom)')
    axs_zoom[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axs_zoom[1].grid(True, linestyle=':')

    pdf.savefig(fig_zoom)
    plt.close(fig_zoom)

    # ---------------------------------------------------------
    # PÁGINA 3: GRÁFICOS (VARREDURA COMPLETA)
    # ---------------------------------------------------------
    fig_full, axs_full = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    fig_full.patch.set_facecolor('#F8F9FA')
    fig_full.suptitle('Resultados Práticos - Varredura Completa (~26k)', fontsize=16, fontweight='bold', color='#2C3E50')

    axs_full[0].plot(lags_completo, norm_antes, color='#8E44AD', linewidth=1)
    axs_full[0].set_title('Correlação Cruzada ANTES (Completa)')
    axs_full[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axs_full[0].grid(True, linestyle=':')

    axs_full[1].plot(lags_completo, norm_depois, color='#27AE60', linewidth=1)
    axs_full[1].set_title('Correlação Cruzada DEPOIS (Completa)')
    axs_full[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axs_full[1].grid(True, linestyle=':')

    pdf.savefig(fig_full)
    plt.close(fig_full)

print(f"Arquivo '{nome_arquivo}' gerado com sucesso!")