# TR3 - Modelos Probabilísticos (IME)

Códigos em Python para a análise estocástica de sinais de vídeo codificados (H.263 e MPEG-4).

## Estrutura das Pastas

- `dados/`: colocar os arquivos `.txt` aqui.
- `resultados/`: pasta criada automaticamente ao rodar os scripts:
  - `passo1_sinais_e_histogramas/`: gráficos no tempo e histogramas com a Gaussiana sobreposta (subpasta `extras/` com comparativos).
  - `passo2_valores_esperados/`: arquivo `estatisticas.txt` com média, variância, desvio padrão, VMQ e RMS.
  - `passo3_facs/`: autocorrelações e correlação cruzada normalizadas e com lags positivos (subpasta `extras/` com 3 em 1 e verificação da fatia).
  - `passo4_descorrelacao/`: memória de cálculo da covariância e curvas de correlação antes e depois (subpasta `extras/` com dispersão).
  - `passo6_7_dep_e_espectro/`: gráficos de DEP em dB, análise da cor do espectro vs ruídos de referência e resumo em `analise_espectro.txt`.

## Como Rodar

1. Instalar as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Colocar os arquivos `.txt` dentro de `dados/`.

3. Executar todos os passos:
   ```bash
   python executar_tudo.py
   ```
   Ou rodar cada script separadamente:
   ```bash
   python passo1_2_sinais_e_momentos.py
   python passo3_facs.py
   python passo4_descorrelacao.py
   python passo6_7_dep_e_espectro.py
   ```
