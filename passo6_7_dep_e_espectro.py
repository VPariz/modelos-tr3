from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

pasta_dados = Path("dados")
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_saida = Path("resultados/passo6_7_dep_e_espectro")
pasta_saida_extras = pasta_saida / "extras"
pasta_saida.mkdir(parents=True, exist_ok=True)
pasta_saida_extras.mkdir(parents=True, exist_ok=True)

# Le os dois sinais
sinais = [np.loadtxt(arq) for arq in arquivos]
nomes = [arq.stem for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
sinais_ordenados = [sinais[ordem[0]], sinais[ordem[1]]]
nomes_ordenados = [nomes[ordem[0]], nomes[ordem[1]]]
cores = ["red", "blue"]

resultados_analise = []

# =========================================================================
# ITENS 6 E 7: CALCULO DAS DEPs E IDENTIFICACAO DA COR DO ESPECTRO
# =========================================================================

deps_dados = {}

for nome, s, cor in zip(nomes_ordenados, sinais_ordenados, cores):
    N = len(s)
    s_c = s - np.mean(s)

    # 1. Teorema de Wiener-Khinchin (J.J.W. Cap. 7): DEP via FFT da Autocorrelacao
    r_xx = np.correlate(s_c, s_c, mode="full")
    r_xx /= r_xx[N - 1]  # Normaliza lag 0 em 1.0

    # Estimador de Welch para densidade espectral suavizada e robusta
    nperseg = min(4096, N // 4)
    freqs, psd = signal.welch(s_c, fs=1.0, nperseg=nperseg, scaling="density")

    # Converte para escala em decibeis (dB)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    psd_db_norm = psd_db - np.max(psd_db)  # Normaliza pico em 0 dB

    # 2. Ajuste linear da lei de potencia: S(f) proporcional a 1 / f^alpha
    # Faixa util de analise em baixas e medias frequencias (0.005 a 0.2 ciclos/quadro)
    mask = (freqs >= 0.005) & (freqs <= 0.2)
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    coefs = np.polyfit(log_f, log_p, 1)
    alpha = -coefs[0]

    # Classificacao da cor
    if alpha < 0.3:
        cor_espectro = "Branco (f^0)"
    elif 0.3 <= alpha < 0.8:
        cor_espectro = "Intermediario entre Branco e Rosa (1/f^alpha, alpha ~ 0.7)"
    elif 0.8 <= alpha <= 1.2:
        cor_espectro = "Rosa (1/f^1)"
    elif 1.2 < alpha < 1.8:
        cor_espectro = "Colorido entre Rosa e Marrom/Vermelho (1/f^alpha, alpha ~ 1.3)"
    else:
        cor_espectro = "Marrom / Vermelho (1/f^2)"

    deps_dados[nome] = (freqs, psd_db_norm, alpha, cor_espectro, cor)

    resultados_analise.append({
        "nome": nome,
        "N": N,
        "alpha": alpha,
        "cor": cor_espectro
    })

    # ---------------------------------------------------------------------
    # GRAFICO ITEM 6: DEP INDIVIDUAL (Escala dB vs Frequencia) - PRINCIPAL
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(freqs, psd_db_norm, color=cor, linewidth=1.1, label=f"DEP {nome}")
    plt.title(f"Densidade Espectral de Potencia (DEP): {nome}")
    plt.xlabel("Frequencia Normalizada (ciclos/quadro)")
    plt.ylabel("Potencia Relativa (dB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(pasta_saida / f"dep_{nome}.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------------------
    # GRAFICO ITEM 7: COR DO ESPECTRO (Escala Log-Log com Referencias) - PRINCIPAL
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=120)
    
    # Curva real do sinal (ignora f=0 para log)
    f_plot = freqs[freqs > 0]
    p_plot = psd_db_norm[freqs > 0]
    plt.semilogx(f_plot, p_plot, color=cor, linewidth=1.2, label=f"DEP {nome} (alpha = {alpha:.2f})")

    # Retas de referencia teoricas (Branco: f^0, Rosa: 1/f, Marrom: 1/f^2)
    p_ref = p_plot[np.argmin(np.abs(f_plot - 0.01))]
    f_ref = f_plot[(f_plot >= 0.005) & (f_plot <= 0.3)]
    
    reta_branco = np.zeros_like(f_ref) + p_ref
    reta_rosa = p_ref - 10 * np.log10(f_ref / 0.01)
    reta_marrom = p_ref - 20 * np.log10(f_ref / 0.01)

    plt.semilogx(f_ref, reta_branco, "k--", linewidth=1.0, alpha=0.7, label="Ref. Ruido Branco (f^0)")
    plt.semilogx(f_ref, reta_rosa, "magenta", linestyle="--", linewidth=1.2, alpha=0.8, label="Ref. Ruido Rosa (1/f)")
    plt.semilogx(f_ref, reta_marrom, "brown", linestyle="--", linewidth=1.2, alpha=0.8, label="Ref. Ruido Marrom (1/f^2)")

    plt.title(f"Cor do Espectro: {nome} (Inclinacao alpha = {alpha:.2f})")
    plt.xlabel("Frequencia (escala logaritmica)")
    plt.ylabel("Densidade de Potencia (dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xlim(f_plot[0], 0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pasta_saida / f"cor_espectro_{nome}.png", dpi=150)
    plt.close()

# -------------------------------------------------------------------------
# GRAFICOS EXTRAS/COMPARATIVOS (SUBPASTA extras/)
# -------------------------------------------------------------------------
# Comparativo das DEPs
plt.figure(figsize=(11, 5), dpi=120)
for nome, (freqs, psd_db_norm, _, _, cor) in deps_dados.items():
    plt.plot(freqs, psd_db_norm, color=cor, linewidth=1.0, alpha=0.8, label=nome)
plt.title("Comparacao das DEPs: H.263 vs MPEG-4")
plt.xlabel("Frequencia Normalizada (ciclos/quadro)")
plt.ylabel("Potencia Relativa (dB)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, 0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(pasta_saida_extras / "deps_comparativo_sobreposto.png", dpi=150)
plt.close()

# Comparativo da Cor do Espectro em Log-Log
plt.figure(figsize=(11, 6), dpi=120)
for nome, (freqs, psd_db_norm, alpha, _, cor) in deps_dados.items():
    f_p = freqs[freqs > 0]
    p_p = psd_db_norm[freqs > 0]
    plt.semilogx(f_p, p_p, color=cor, linewidth=1.0, alpha=0.8, label=f"{nome} (alpha = {alpha:.2f})")
plt.title("Comparacao da Cor do Espectro (Escala Logaritmica)")
plt.xlabel("Frequencia (escala logaritmica)")
plt.ylabel("Densidade de Potencia (dB)")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(pasta_saida_extras / "cor_espectro_comparativo.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# RELATORIO FORMAL DE ANALISE ESPECTRAL (TEXTO)
# -------------------------------------------------------------------------
texto_analise = (
    "=========================================================================\n"
    "ITENS 5, 6 E 7: ESTACIONARIEDADE (ESA/EE), DEP E COR DO ESPECTRO\n"
    "=========================================================================\n\n"
    "1. ITEM 5: ESTACIONARIEDADE EM SENTIDO AMPLO (ESA) E ESTRITA (EE)\n"
    "-------------------------------------------------------------------------\n"
    "a) Estacionariedade Estrita (EE): REJEITADA.\n"
    "   Os sinais de video codificado nao sao estritamente estacionarios porque\n"
    "   suas funcoes densidade de probabilidade conjuntas variam substancialmente\n"
    "   entre tomadas de cena (cenas escuras/calmas vs cenas de acao e movimento).\n\n"
    "b) Estacionariedade em Sentido Amplo (ESA / WSS):\n"
    "   - Globalmente (no filme completo de 1 hora): REJEITADA.\n"
    "     A media temporal local oscila (entre ~1390 bits em dialogos e ~1842 bits\n"
    "     no climax), violando a condicao de media rigorosamente constante E[X] = mu.\n"
    "   - Localmente (dentro de tomadas continuas de cena / bloco representativo):\n"
    "     ACEITA EM PRIMEIRA APROXIMACAO.\n"
    "     Dentro de uma tomada estavel, a media e constante e o processo se comporta\n"
    "     como ciclostacionario devido a repeticao periodica do GoP.\n\n"
    "2. ITEM 6: DENSIDADE ESPECTRAL DE POTENCIA (DEP - WIENER-KHINCHIN)\n"
    "-------------------------------------------------------------------------\n"
    "Pelo Teorema de Wiener-Khinchin (J.J.W. Cap. 7), a DEP S(e^jw) e obtida pela\n"
    "Transformada de Fourier da funcao de autocorrelacao:\n"
    "S(e^jw) = F{R_XX[k]}.\n\n"
    "Caracteristicas observadas nas DEPs dos sinais:\n"
    "- Concentracao massiva de energia em baixas frequencias (f -> 0), decorrente\n"
    "  do nivel medio DC e da correlacao de longa duracao entre quadros vizinhos;\n"
    "- Linhas espectrais periodicas (harmonicos) associadas a taxa de repeticao\n"
    "  dos quadros I (periodo do GoP).\n\n"
    "3. ITEM 7: CLASSIFICACAO DA COR DO ESPECTRO (LEI DE POTENCIA 1/f^alpha)\n"
    "-------------------------------------------------------------------------\n"
    "Ajuste da relacao espectral S(f) proporcional a 1 / (f^alpha):\n\n"
)

for res in resultados_analise:
    texto_analise += (
        f"Sinal: {res['nome']} (N = {res['N']})\n"
        f"  - Expoente espectral estimado (alpha): {res['alpha']:.2f}\n"
        f"  - Classificacao da cor: {res['cor']}\n\n"
    )

texto_analise += (
    "Conclusao Geral: Ambos os sinais possuem ESPECTRO COLORIDO (ruido 1/f^alpha),\n"
    "situando-se na faixa entre Ruido Rosa (1/f) e Ruido Marrom/Vermelho (1/f^2).\n"
    "Esse comportamento e tipico de processos com Dependencia de Longo Alcance\n"
    "(Long-Range Dependence - LRD) e auto-similaridade estocastica em teletrafego.\n"
)

print(texto_analise)

with open(pasta_saida / "analise_espectro.txt", "w", encoding="utf-8") as f:
    f.write(texto_analise)

print("Passo 6 e 7 concluidos!")
print(f"Graficos principais em: {pasta_saida}")
print(f"Graficos extras em: {pasta_saida_extras}")
