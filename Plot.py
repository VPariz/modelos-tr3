from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

pasta = Path("dados")
arquivos = sorted(pasta.glob("*.txt"))
cores = ["red", "blue"]

# Cria a pasta de resultados (se já existir, não dá erro)
pasta_resultados = Path("resultados")
pasta_resultados.mkdir(parents=True, exist_ok=True)

dados = {}

# Abre o arquivo de texto na pasta 'resultados' para gravar os valores
with open(pasta_resultados / "estatisticas.txt", "w", encoding="utf-8") as f_txt:

    # Loop dos arquivos
    for arquivo, cor in zip(arquivos, cores):
        with open(arquivo, "r", encoding="utf-8") as f:
            valores = [float(linha.strip()) for linha in f if linha.strip()]

        # Converte para array numpy para facilitar as contas
        arr = np.array(valores)

        # Cálculos estatísticos
        media = np.mean(arr)
        variancia = np.var(arr)
        desvio_padrao = np.std(arr)
        vmq = np.mean(arr**2)
        rms = np.sqrt(vmq)

        # Exibe os resultados no terminal
        print(f"\nValores Esperados: {arquivo.name}")
        print(f"Média:          {media:.4f}")
        print(f"Variância:      {variancia:.4f}")
        print(f"Desvio Padrão:  {desvio_padrao:.4f}")
        print(f"VMQ:            {vmq:.4f}")
        print(f"RMS:            {rms:.4f}")

        # Salva os mesmos resultados no arquivo TXT
        f_txt.write(f"Valores Esperados: {arquivo.name}\n")
        f_txt.write(f"Média:          {media:.4f}\n")
        f_txt.write(f"Variância:      {variancia:.4f}\n")
        f_txt.write(f"Desvio Padrão:  {desvio_padrao:.4f}\n")
        f_txt.write(f"VMQ:            {vmq:.4f}\n")
        f_txt.write(f"RMS:            {rms:.4f}\n")
        f_txt.write("-" * 40 + "\n\n")

        dados[arquivo.name] = (valores, cor)

        # Gráfico individual (maior, linha fina e sem borrão)
        plt.figure(figsize=(15, 6), dpi=120)
        plt.plot(valores, color=cor, linewidth=0.3, alpha=0.5, label=arquivo.name)
        plt.title(f"Gráfico Individual: {arquivo.name}")
        plt.xlabel("Amostra")
        plt.ylabel("Valor")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_resultados / f"grafico_{arquivo.stem}.png", dpi=150)
        plt.close()

        # Histograma
        bins = 120 if "h263" in arquivo.name else 150
        plt.figure(figsize=(10, 5), dpi=120)
        plt.hist(arr, bins=bins, density=True, color=cor, alpha=0.6, label="Histograma")
        plt.title(f"Histograma: {arquivo.name}")
        plt.xlabel("Valor")
        plt.ylabel("Densidade")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pasta_resultados / f"histograma_{arquivo.stem}.png", dpi=150)
        plt.close()

        # Histograma com a Gaussiana teórica
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
        plt.savefig(pasta_resultados / f"histograma_gaussiana_{arquivo.stem}.png", dpi=150)
        plt.close()

# --- PARTE 2: Gráfico Comparativo ---
plt.figure(figsize=(15, 6), dpi=120)

for nome, (valores, cor) in dados.items():
    plt.plot(valores, color=cor, linewidth=0.3, alpha=0.5, label=nome)

plt.title("Comparação: Arquivo 1 vs Arquivo 2")
plt.xlabel("Amostra")
plt.ylabel("Valor")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Salva o gráfico comparativo na pasta resultados
plt.savefig(pasta_resultados / "grafico_comparativo.png", dpi=150)
plt.close()