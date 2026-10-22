# Trabalho 3 - Modelos Probabilísticos (IME)

Códigos em Python para o TR3 de Modelos Probabilísticos (Análise Estocástica de Sinais de Vídeo Codificados).

## Estrutura do Projeto

- `dados/`: pasta para os arquivos de entrada `.txt`.
- `resultados/`: pasta gerada automaticamente ao rodar os scripts, organizada pelos passos da professora:
  - `passo1_sinais_e_histogramas/`: gráficos temporais individuais, comparativo e histogramas com a Gaussiana teórica.
  - `passo2_valores_esperados/`: arquivo `estatisticas.txt` com média, variância, desvio padrão, VMQ e RMS.
  - `passo3_facs/`: as 3 FACs (autocorrelação de cada sinal e correlação cruzada, completas e com zoom).
  - `passo4_descorrelacao/`: matrizes de covariância antes e depois, autovalores, autovetores e gráfico de dispersão.

## Scripts

- `passo1_2_sinais_e_momentos.py`: resolve os itens 1 e 2 do roteiro.
- `passo3_facs.py`: resolve o item 3 (as 3 FACs).
- `passo4_descorrelacao.py`: resolve o item 4 (descorrelação 2-a-2 via diagonalização da matriz de covariância - JJW Cap. 6).
- `executar_tudo.py`: roda todos os passos em sequência.

## Como Executar

1. Instalar as bibliotecas necessárias:
   ```bash
   pip install -r requirements.txt
   ```

2. Colocar os arquivos `.txt` dentro da pasta `dados/`.

3. Executar o passo desejado ou rodar tudo de uma vez:
   ```bash
   python executar_tudo.py
   ```
