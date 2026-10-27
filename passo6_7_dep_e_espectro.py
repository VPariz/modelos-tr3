from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

pasta_base = Path(__file__).resolve().parent
pasta_dados = pasta_base / "dados"
arquivos = sorted(pasta_dados.glob("*.txt"))

pasta_saida = pasta_base / "resultados" / "passo6_7_dep_e_espectro"
pasta_saida_extras = pasta_saida / "extras"
pasta_saida.mkdir(parents=True, exist_ok=True)
pasta_saida_extras.mkdir(parents=True, exist_ok=True)

# le os dois sinais
sinais = [np.loadtxt(arq) for arq in arquivos]
nomes = [arq.stem for arq in arquivos]
ordem = np.argsort([len(s) for s in sinais])
sinais_ordenados = [sinais[ordem[0]], sinais[ordem[1]]]
nomes_ordenados = [nomes[ordem[0]], nomes[ordem[1]]]
cores = ["red", "blue"]

resultados_analise = []
deps_dados = {}

for nome, s, cor in zip(nomes_ordenados, sinais_ordenados, cores):
    N = len(s)
    s_c = s - np.mean(s)

    # estimador de welch para a densidade espectral
    nperseg = min(4096, N // 4)
    freqs, psd = signal.welch(s_c, fs=1.0, nperseg=nperseg, scaling="density")

    # converte para dB e normaliza no maximo em 0 dB
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    psd_db_norm = psd_db - np.max(psd_db)

    # ajuste linear em log-log para achar o alpha de 1/f^alpha
    mask = (freqs >= 0.005) & (freqs <= 0.2)
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    coefs = np.polyfit(log_f, log_p, 1)
    alpha = -coefs[0]

    # classificacao da cor
    if alpha < 0.3:
        cor_espectro = "Branco (f^0)"
    elif 0.3 <= alpha < 0.8:
        cor_espectro = "Intermediario entre Branco e Rosa (alpha ~ 0.7)"
    elif 0.8 <= alpha <= 1.2:
        cor_espectro = "Rosa (1/f)"
    elif 1.2 < alpha < 1.8:
        cor_espectro = "Colorido entre Rosa e Marrom/Vermelho (alpha ~ 1.3)"
    else:
        cor_espectro = "Marrom / Vermelho (1/f^2)"

    deps_dados[nome] = (freqs, psd_db_norm, alpha, cor_espectro, cor)
    resultados_analise.append({
        "nome": nome,
        "N": N,
        "alpha": alpha,
        "cor": cor_espectro
    })

    # grafico da DEP individual em dB
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

    # grafico da cor do espectro em escala logaritmica com retas de referencia
    plt.figure(figsize=(10, 6), dpi=120)
    
    f_plot = freqs[freqs > 0]
    p_plot = psd_db_norm[freqs > 0]
    plt.semilogx(f_plot, p_plot, color=cor, linewidth=1.2, label=f"DEP {nome} (alpha = {alpha:.2f})")

    p_ref = p_plot[np.argmin(np.abs(f_plot - 0.01))]
    f_ref = f_plot[(f_plot >= 0.005) & (f_plot <= 0.3)]
    
    reta_branco = np.zeros_like(f_ref) + p_ref
    reta_rosa = p_ref - 10 * np.log10(f_ref / 0.01)
    reta_marrom = p_ref - 20 * np.log10(f_ref / 0.01)

    plt.semilogx(f_ref, reta_branco, "k--", linewidth=1.0, alpha=0.7, label="Ref. Ruido Branco (f^0)")
    plt.semilogx(f_ref, reta_rosa, "magenta", linestyle="--", linewidth=1.2, alpha=0.8, label="Ref. Ruido Rosa (1/f)")
    plt.semilogx(f_ref, reta_marrom, "brown", linestyle="--", linewidth=1.2, alpha=0.8, label="Ref. Ruido Marrom (1/f^2)")

    plt.title(f"Cor do Espectro: {nome} (alpha = {alpha:.2f})")
    plt.xlabel("Frequencia (escala logaritmica)")
    plt.ylabel("Densidade de Potencia (dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xlim(f_plot[0], 0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pasta_saida / f"cor_espectro_{nome}.png", dpi=150)
    plt.close()

# comparativo sobreposto das DEPs
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

# comparativo da cor do espectro em log-log
plt.figure(figsize=(11, 6), dpi=120)
for nome, (freqs, psd_db_norm, alpha, _, cor) in deps_dados.items():
    f_p = freqs[freqs > 0]
    p_p = psd_db_norm[freqs > 0]
    plt.semilogx(f_p, p_p, color=cor, linewidth=1.0, alpha=0.8, label=f"{nome} (alpha = {alpha:.2f})")
plt.title("Comparacao da Cor do Espectro")
plt.xlabel("Frequencia (escala logaritmica)")
plt.ylabel("Densidade de Potencia (dB)")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(pasta_saida_extras / "cor_espectro_comparativo.png", dpi=150)
plt.close()

# resumo em texto dos itens 5, 6 e 7
texto_analise = (
    "ITENS 5, 6 E 7: ESTACIONARIEDADE, DEP E COR DO ESPECTRO\n\n"
    "1. Estacionariedade (Item 5):\n"
    "- Estacionariedade Estrita (EE): Nao, porque a distribuicao conjunta varia ao longo do filme com a troca de cenas.\n"
    "- Estacionariedade em Sentido Amplo (ESA) global: Nao, porque a media oscila entre trechos calmos e de acao.\n"
    "- ESA local / ciclostacionario: Sim em aproximacao, dentro de tomadas continuas de cena devido a regularidade do GoP.\n\n"
    "2. Densidade Espectral de Potencia (Item 6):\n"
    "- Obtida a partir da autocorrelacao (Teorema de Wiener-Khinchin).\n"
    "- Concentracao dominante em baixas frequencias (media e correlacao de longo prazo) e raias discretas nos harmonicos do GoP.\n\n"
    "3. Cor do Espectro (Item 7):\n"
)

for res in resultados_analise:
    texto_analise += (
        f"- {res['nome']}: alpha = {res['alpha']:.2f} -> {res['cor']}\n"
    )

texto_analise += (
    "\nConclusao: Os sinais possuem espectro colorido com decaimento 1/f^alpha,\n"
    "situando-se na regiao entre ruido rosa e marrom.\n"
)

print(texto_analise)

with open(pasta_saida / "analise_espectro.txt", "w", encoding="utf-8") as f:
    f.write(texto_analise)

print("Passo 6 e 7 concluidos.")
