from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

pasta_dados = Path("dados")
arquivos = sorted(pasta_dados.glob("*.txt"))
cores = ["red", "blue"]

# Pastas de saída organizadas por passo
pasta_passo1 = Path("resultados/passo1_sinais_e_histogramas")
pasta_passo2 = Path("resultados/passo2_valores_esperados")
pasta_passo1.mkdir(parents=True, exist_ok=True)
pasta_passo2.mkdir(parents=True, exist_ok=True)

dados = {}

# Salva as estatísticas no passo 2
with open(pasta_passo2 / "estatisticas.txt", "w", encoding="utf-8") as f_txt:

    for arquivo, cor in zip(arquivos, cores):
        with open(arquivo, "r", encoding="utf-8") as f:
            valores = [float(linha.strip()) for linha in f if linha.strip()]

        arr = np.array(valores)

        # Cálculos estatísticos (Passo 2)
        media = np.mean(arr)
        variancia = np.var(arr)
        desvio_padrao = np.std(arr)
        vmq = np.mean(arr**2)
        rms = np.sqrt(vmq)

        # Exibe no terminal
        print(f"\nValores Esperados: {arquivo.name}")
        print(f"Média:          {media:.4f}")
        print(f"Variância:      {variancia:.4f}")
        print(f"Desvio Padrão:  {desvio_padrao:.4f}")
        print(f"VMQ:            {vmq:.4f}")
        print(f"RMS:            {rms:.4f}")

        f_txt.write(f"Valores Esperados: {arquivo.name}\n")
        f_txt.write(f"Média:          {media:.4f}\n")
        f_txt.write(f"Variância:      {variancia:.4f}\n")
        f_txt.write(f"Desvio Padrão:  {desvio_padrao:.4f}\n")
        f_txt.write(f"VMQ:            {vmq:.4f}\n")
        f_txt.write(f"RMS:            {rms:.4f}\n")
        f_txt.write("-" * 40 + "\n\n")

        dados[arquivo.name] = (valores, cor)

        # Gráfico individual no tempo (Passo 1)
        plt.figure(figsize=(15, 6), dpi=120)
        plt.plot(valores, color=cor, linewidth=0.3, alpha=0.5, label=arquivo.name)
        plt.title(f"Gráfico Individual: {arquivo.name}")
        plt.xlabel("Amostra")
        plt.ylabel("Valor")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_passo1 / f"grafico_{arquivo.stem}.png", dpi=150)
        plt.close()

        # Histograma normalizado (Passo 1)
        bins = 120 if "h263" in arquivo.name else 150
        plt.figure(figsize=(10, 5), dpi=120)
        plt.hist(arr, bins=bins, density=True, color=cor, alpha=0.6, label="Histograma")
        plt.title(f"Histograma: {arquivo.name}")
        plt.xlabel("Valor")
        plt.ylabel("Densidade")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_passo1 / f"histograma_{arquivo.stem}.png", dpi=150)
        plt.close()

        # Histograma com Gaussiana Teórica (Passo 1)
        plt.figure(figsize=(10, 5), dpi=120)
        plt.hist(arr, bins=bins, density=True, color=cor, alpha=0.6, label="Histograma")

        x = np.linspace(0, np.max(arr), 1000)
        gaussiana = norm.pdf(x, media, desvio_padrao)
        plt.plot(x, gaussiana, color="black", linestyle="--", linewidth=1.8, label="Gaussiana teórica")

        plt.title(f"Histograma vs Gaussiana: {arquivo.name}")
        plt.xlabel("Valor")
        plt.ylabel("Densidade")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_passo1 / f"histograma_gaussiana_{arquivo.stem}.png", dpi=150)
        plt.close()

# Gráfico Comparativo no Tempo (Passo 1)
plt.figure(figsize=(15, 6), dpi=120)
for nome, (valores, cor) in dados.items():
    plt.plot(valores, color=cor, linewidth=0.3, alpha=0.5, label=nome)

plt.title("Comparação: Arquivo 1 vs Arquivo 2")
plt.xlabel("Amostra")
plt.ylabel("Valor")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(pasta_passo1 / "grafico_comparativo.png", dpi=150)
plt.close()

print("\nPasso 1 e 2 concluídos!")
print(f"Gráficos salvos em: {pasta_passo1}")
print(f"Estatísticas salvas em: {pasta_passo2 / 'estatisticas.txt'}")
