import subprocess
import sys

scripts = [
    ("Passo 1 e 2 (Sinais, Histogramas e Valores Esperados)", "passo1_2_sinais_e_momentos.py"),
    ("Passo 3 (Funcoes de Autocorrelacao e Correlacao Cruzada)", "passo3_facs.py"),
    ("Passo 4 (Descorrelacao 2-a-2 via Diagonalizacao)", "passo4_descorrelacao.py"),
]

print("=" * 60)
print("EXECUTANDO TODOS OS PASSOS DO TR3 (MODELOS PROBABILISTICOS)")
print("=" * 60)

for descricao, script in scripts:
    print(f"\n>>> Rodando: {descricao} ({script})...")
    res = subprocess.run([sys.executable, script])
    if res.returncode != 0:
        print(f"Erro ao executar {script}!")
        sys.exit(res.returncode)

print("\n" + "=" * 60)
print("TODOS OS PASSOS FORAM EXECUTADOS COM SUCESSO!")
print("Estrutura da pasta 'resultados/':")
print("  |-- passo1_sinais_e_histogramas/")
print("  |     |-- (graficos individuais do relatorio)")
print("  |     +-- extras/ (comparativos e sobrepostos)")
print("  |-- passo2_valores_esperados/")
print("  |     +-- estatisticas.txt")
print("  |-- passo3_facs/")
print("  |     |-- (FACs individuais e comparacao de valores esperados)")
print("  |     +-- extras/ (3-em-1 e histograma sobreposto)")
print("  |-- passo4_descorrelacao/")
print("        |-- descorrelacao.txt e correlacao antes/depois")
print("        +-- extras/ (dispersao e zoom simetrico)")
print("=" * 60)
