from pathlib import Path
import subprocess
import sys

scripts = [
    ("Passo 1 e 2 (Sinais, Histogramas e Momentos)", "passo1_2_sinais_e_momentos.py"),
    ("Passo 3 (Funcoes de Autocorrelacao e Cruzada)", "passo3_facs.py"),
    ("Passo 4 (Descorrelacao 2-a-2)", "passo4_descorrelacao.py"),
    ("Passo 6 e 7 (DEPs e Cor do Espectro)", "passo6_7_dep_e_espectro.py"),
]

print("-" * 50)
print("Executando scripts do TR3 de Modelos Probabilisticos")
print("-" * 50)

for descricao, script in scripts:
    print(f"\nRodando: {descricao} ({script})...")
    res = subprocess.run([sys.executable, script])
    if res.returncode != 0:
        print(f"Erro ao executar {script}!")
        sys.exit(res.returncode)

print("\n" + "-" * 50)
print("Todos os passos foram executados com sucesso.")
print("Resultados salvos em 'resultados/'.")
print("-" * 50)
