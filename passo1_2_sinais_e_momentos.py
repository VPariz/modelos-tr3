from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skew, kurtosis

pasta_base = Path(__file__).resolve().parent
pasta_dados = pasta_base / "dados"
arquivos = sorted(pasta_dados.glob("*.txt"))
cores = ["red", "blue"]

pasta_passo1 = pasta_base / "resultados" / "passo1_sinais_e_histogramas"
pasta_passo1_extras = pasta_passo1 / "extras"
pasta_passo2 = pasta_base / "resultados" / "passo2_valores_esperados"

pasta_passo1.mkdir(parents=True, exist_ok=True)
pasta_passo1_extras.mkdir(parents=True, exist_ok=True)
pasta_passo2.mkdir(parents=True, exist_ok=True)

dados = {}
arrays = {}

# salva estatisticas pedidas no item 2
with open(pasta_passo2 / "estatisticas.txt", "w", encoding="utf-8") as f_txt:
    for arquivo, cor in zip(arquivos, cores):
        with open(arquivo, "r", encoding="utf-8") as f:
            valores = [float(linha.strip()) for linha in f if linha.strip()]

        arr = np.array(valores)
        arrays[arquivo.name] = arr

        # media, variancia, desvio, vmq, rms, assimetria e curtose
        media = np.mean(arr)
        variancia = np.var(arr)
        desvio_padrao = np.std(arr)
        vmq = np.mean(arr**2)
        rms = np.sqrt(vmq)
        assimetria = skew(arr)
        curtose_pearson = kurtosis(arr, fisher=False)

        print(f"\n{arquivo.name} (N = {len(arr)}):")
        print(f"  Media:              {media:.4f}")
        print(f"  Variancia:          {variancia:.4f}")
        print(f"  Desvio Padrao:      {desvio_padrao:.4f}")
        print(f"  VMQ E[X^2]:         {vmq:.4f}")
        print(f"  RMS sqrt(VMQ):      {rms:.4f}")
        print(f"  Assimetria (Skew):  {assimetria:.4f}")
        print(f"  Curtose (Pearson):  {curtose_pearson:.4f} (Gaussiana = 3.0)")

        f_txt.write(f"Arquivo: {arquivo.name} (N = {len(arr)} amostras)\n")
        f_txt.write(f"Media E[X]:         {media:.4f}\n")
        f_txt.write(f"Variancia Var(X):   {variancia:.4f}\n")
        f_txt.write(f"Desvio Padrao:      {desvio_padrao:.4f}\n")
        f_txt.write(f"VMQ E[X^2]:         {vmq:.4f}\n")
        f_txt.write(f"RMS sqrt(E[X^2]):   {rms:.4f}\n")
        f_txt.write(f"Assimetria (Skew):  {assimetria:.4f}\n")
        f_txt.write(f"Curtose (Pearson):  {curtose_pearson:.4f} (Ref. Gaussiana = 3.0000)\n")
        f_txt.write("-" * 48 + "\n\n")

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

    # Momentos conjuntos / cruzados entre as duas sequencias (com bloco representativo N = 26192)
    nomes_arq = list(arrays.keys())
    if len(nomes_arq) == 2:
        x_h263 = arrays[nomes_arq[0]]
        y_mpeg_full = arrays[nomes_arq[1]]
        N = len(x_h263)
        # Bloco representativo otimo (k = 8992 a 35184)
        k_ini = 8992
        y_bloco = y_mpeg_full[k_ini : k_ini + N]

        correlacao_xy = np.mean(x_h263 * y_bloco)
        cov_xy = np.mean((x_h263 - np.mean(x_h263)) * (y_bloco - np.mean(y_bloco)))
        rho_xy = cov_xy / (np.std(x_h263) * np.std(y_bloco))
        vmq_diff = np.mean((x_h263 - y_bloco) ** 2)

        f_txt.write("ESTATISTICAS CONJUNTAS ENTRE AS DUAS SEQUENCIAS (N = 26192)\n")
        f_txt.write(f"(Bloco MPEG-4 representativo: amostras {k_ini} a {k_ini + N})\n")
        f_txt.write(f"Correlacao Cruzada R_XY(0) = E[X*Y]:       {correlacao_xy:.4f}\n")
        f_txt.write(f"Covariancia Cruzada K_XY = Cov(X,Y):       {cov_xy:.4f}\n")
        f_txt.write(f"Coeficiente de Correlacao rho_XY:          {rho_xy:.6f}\n")
        f_txt.write(f"VMQ da Diferenca E[(X - Y)^2]:             {vmq_diff:.4f}\n")
        f_txt.write("-" * 48 + "\n")

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
