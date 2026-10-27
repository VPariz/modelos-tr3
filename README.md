# Análise Estocástica e Espectral de Sinais de Vídeo Codificados

**Instituto Militar de Engenharia — IME**  
**Área:** Processamento Estatístico de Sinais e Processos Estocásticos  
**Padrões Analisados:** Fluxos de vídeo codificados em **H.263** e **MPEG-4**  

---

## Objetivo do Projeto

Investigação estocástica rigorosa do comportamento estatístico e espectral de sequências de quadros de vídeo digital (tamanho de frames em bits ao longo do tempo), aplicando:

1. **Estatística Descritiva & Momentos:** Cálculo de Média ($\mu$), Variância ($\sigma^2$), Desvio Padrão ($\sigma$), Valor Médio Quadrático (VMQ) e Valor Eficaz (RMS).
2. **Aderência Gaussiana:** Geração de histogramas de distribuição empírica com sobreposição da curva Normal teórica.
3. **Funções de Correlação:** Determinação da Função de Autocorrelação (FAC) unilateral normalizada e Função de Correlação Cruzada (FCC) com lags positivos ($k \ge 0$).
4. **Descorrelação & Covariância:** Cálculo da matriz de covariância entre canais e verificação das curvas de correlação antes e após a transformação de descorrelação linear.
5. **Densidade Espectral de Potência (DEP):** Estimativa espectral em dB via Transformada Rápida de Fourier (FFT / Periodograma) e classificação cromática de ruído (ruído branco vs ruído colorido/rosa/marrom).

---

## Estrutura Modular

```
modelos-tr3/
├── dados/                               # Arquivos de dados brutos (.txt) com sequências temporais
├── resultados/                          # Saídas analíticas e figuras geradas automaticamente
│   ├── passo1_sinais_e_histogramas/     # Gráficos temporais x(n) e histogramas com Gaussiana
│   ├── passo2_valores_esperados/        # Relatório estatístico consolidado (estatisticas.txt)
│   ├── passo3_facs/                     # Autocorrelações (FAC) e correlações cruzadas (FCC)
│   ├── passo4_descorrelacao/            # Matrizes de dispersão, covariância e descorrelação
│   └── passo6_7_dep_e_espectro/         # Gráficos da Densidade Espectral de Potência (DEP em dB)
├── passo1_2_sinais_e_momentos.py        # Módulo de momentos e distribuições
├── passo3_facs.py                       # Módulo de autocorrelação e correlação cruzada
├── passo4_descorrelacao.py              # Módulo de cálculo de matrizes e descorrelação
├── passo6_7_dep_e_espectro.py           # Módulo espectral e classificação cromática
├── executar_tudo.py                     # Pipeline automatizado de execução ponta a ponta
└── requirements.txt                     # Dependências (numpy, scipy, matplotlib)
```

---

## Fundamentação Matemática & Métodos

* **Função de Autocorrelação Estimada:**
  $$R_{xx}[k] = \frac{1}{N - k} \sum_{n=0}^{N-k-1} (x[n] - \mu_x)(x[n+k] - \mu_x)$$

* **Densidade Espectral de Potência (Wiener-Khinchin):**
  $$S_{xx}(f) = \mathcal{F}\{ R_{xx}[k] \} = \sum_{k=-\infty}^{\infty} R_{xx}[k] e^{-j 2 \pi f k}$$

---

## Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Insira os arquivos `.txt` contendo os dados dos quadros em `dados/`.

3. Execute o pipeline completo:
```bash
python executar_tudo.py
```

Os resultados gráficos em alta resolução e as tabelas numéricas serão gerados automaticamente dentro do diretório `resultados/`.
