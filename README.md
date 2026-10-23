# Trabalho 3 - Modelos Probabilísticos (IME)

Códigos em Python para o TR3 de Modelos Probabilísticos (Análise Estocástica de Sinais de Vídeo Codificados).

## Estrutura do Projeto

- `dados/`: pasta para os arquivos de entrada `.txt`.
- `resultados/`: pasta gerada automaticamente ao rodar os scripts, com os gráficos principais na pasta direta e os complementares/sobrepostos em subpastas `extras/`:
  - `passo1_sinais_e_histogramas/`:
    - Principais: gráficos temporais individuais e histogramas com a Gaussiana teórica de cada sinal.
    - `extras/`: sinais e histogramas comparativos sobrepostos.
  - `passo2_valores_esperados/`:
    - `estatisticas.txt`: média, variância, desvio padrão, VMQ e RMS.
  - `passo3_facs/`:
    - Principais: autocorrelações individuais e correlação cruzada (varredura completa e zoom), além da tabela de equivalência do bloco cortado (`comparacao_valores_esperados_bloco.txt`).
    - `extras/`: gráficos combinados 3-em-1 e histograma sobreposto de verificação da PDF.
  - `passo4_descorrelacao/`:
    - Principais: memória de cálculo formal (`descorrelacao.txt`) e curvas de correlação cruzada antes vs depois (zoom e completa).
    - `extras/`: dispersão em nuvem de pontos e zoom simétrico (-250 a +250).
  - `passo6_7_dep_e_espectro/`:
    - Principais: memória formal (`analise_espectro.txt`), DEPs individuais em dB e curvas de classificação da Cor do Espectro vs ruídos de referência.
    - `extras/`: DEPs e Cores do Espectro comparativas sobrepostas.

## Scripts

- `passo1_2_sinais_e_momentos.py`: resolve os itens 1 e 2 do roteiro.
- `passo3_facs.py`: resolve o item 3 (as 3 FACs).
- `passo4_descorrelacao.py`: resolve o item 4 (descorrelação 2-a-2 via diagonalização da matriz de covariância - JJW Cap. 6).
- `passo6_7_dep_e_espectro.py`: resolve os itens 5, 6 e 7 (ESA/EE, DEPs via Wiener-Khinchin e classificação da Cor do Espectro).
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
