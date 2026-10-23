from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

pasta_dados = Path("dados")
arquivos = sorted(pasta_dados.glob("*.txt"))
cores = ["red", "blue"]

pasta_passo1 = Path("resultados/passo1_sinais_e_histogramas")
pasta_passo1_extras = pasta_passo1 / "extras"
pasta_passo2 = Path("resultados/passo2_valores_esperados")

pasta_passo1.mkdir(parents=True, exist_ok=True)
pasta_passo1_extras.mkdir(parents=True, exist_ok=True)
pasta_passo2.mkdir(parents=True, exist_ok=True)

dados = {}

# salva estatisticas pedidas no item 2
with open(pasta_passo2 / "estatisticas.txt", "w", encoding="utf-8") as f_txt:
    for arquivo, cor in zip(arquivos, cores):
        with open(arquivo, "r", encoding="utf-8") as f:
            valores = [float(linha.strip()) for linha in f if linha.strip()]

        arr = np.array(valores)

        # media, variancia, desvio, vmq e rms
        media = np.mean(arr)
        variancia = np.var(arr)
        desvio_padrao = np.std(arr)
        vmq = np.mean(arr**2)
        rms = np.sqrt(vmq)

        print(f"\n{arquivo.name}:")
        print(f"  Media:         {media:.4f}")
        print(f"  Variancia:     {variancia:.4f}")
        print(f"  Desvio Padrao: {desvio_padrao:.4f}")
        print(f"  VMQ:           {vmq:.4f}")
        print(f"  RMS:           {rms:.4f}")

        f_txt.write(f"Arquivo: {arquivo.name}\n")
        f_txt.write(f"Media:         {media:.4f}\n")
        f_txt.write(f"Variancia:     {variancia:.4f}\n")
        f_txt.write(f"Desvio Padrao: {desvio_padrao:.4f}\n")
        f_txt.write(f"VMQ:           {vmq:.4f}\n")
        f_txt.write(f"RMS:           {rms:.4f}\n")
        f_txt.write("-" * 35 + "\n\n")

        dados[arquivo.name] = (valores, cor)

        # grafico no tempo individual
        plt.figure(figsize=(15, 6), dpi=120)
        plt.plot(valores, color=cor, linewidth=0.3, alpha=0.5, label=arquivo.name)
        plt.title(f"Sinal no Tempo: {arquivo.name}")
        plt.xlabel("Amostra")
        plt.ylabel("Valor")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_passo1 / f"sinal_{arquivo.stem}.png", dpi=150)
        plt.close()

        # histograma com a gaussiana teorica sobreposta
        bins = 120 if "h263" in arquivo.name else 150
        plt.figure(figsize=(10, 5), dpi=120)
        plt.hist(arr, bins=bins, density=True, color=cor, alpha=0.6, label="Histograma")

        x = np.linspace(0, np.max(arr), 1000)
        gaussiana = norm.pdf(x, media, desvio_padrao)
        plt.plot(x, gaussiana, color="black", linestyle="--", linewidth=1.8, label="Gaussiana teorica")

        plt.title(f"Histograma vs Gaussiana: {arquivo.name}")
        plt.xlabel("Valor")
        plt.ylabel("Densidade")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_passo1 / f"histograma_gaussiana_{arquivo.stem}.png", dpi=150)
        plt.close()

        # histograma simples sem curva
        plt.figure(figsize=(10, 5), dpi=120)
        plt.hist(arr, bins=bins, density=True, color=cor, alpha=0.6, label="Histograma")
        plt.title(f"Histograma: {arquivo.name}")
        plt.xlabel("Valor")
        plt.ylabel("Densidade")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_passo1_extras / f"histograma_simples_{arquivo.stem}.png", dpi=150)
        plt.close()

# comparativo no tempo sobreposto
plt.figure(figsize=(15, 6), dpi=120)
for nome, (valores, cor) in dados.items():
    plt.plot(valores, color=cor, linewidth=0.3, alpha=0.5, label=nome)

plt.title("Comparacao Temporal: Arquivo 1 vs Arquivo 2")
plt.xlabel("Amostra")
plt.ylabel("Valor")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(pasta_passo1_extras / "sinais_comparativo_sobreposto.png", dpi=150)
plt.close()

print("\nPasso 1 e 2 concluidos.")
