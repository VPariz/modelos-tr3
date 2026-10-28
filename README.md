# Stochastic & Spectral Analysis of Encoded Video Signals

[![Build & CI](https://github.com/VPariz/modelos-tr3/actions/workflows/ci.yml/badge.svg)](https://github.com/VPariz/modelos-tr3/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=black)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)](https://matplotlib.org/)
[![Institution](https://img.shields.io/badge/IME-Electronics_Engineering-104911?style=flat-square)](https://ime.eb.mil.br/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**Instituto Militar de Engenharia — IME**  
**Field:** Statistical Signal Processing & Stochastic Systems  
**Standards Investigated:** Digital video frame streams encoded under **H.263** and **MPEG-4**

---

## 🎯 Project Overview

Rigorous stochastic and spectral characterization of digital video transmission streams (frame size in bits over discrete time series), investigating:

1. **Descriptive Statistics & Higher-Order Moments:** Computation of Sample Mean ($\mu$), Variance ($\sigma^2$), Standard Deviation ($\sigma$), Mean Squared Value (MSV), and Root Mean Square (RMS).
2. **Empirical Distribution & Normality Testing:** Histogram generation with theoretical Gaussian / Normal density overlays.
3. **Correlation Analysis:** One-sided normalized Autocorrelation Functions (ACF) and Cross-Correlation Functions (CCF) across discrete positive lags ($k \ge 0$).
4. **Covariance & Signal Decorrelation:** Channel covariance matrix formulation, scatter visualization, and linear decorrelation transforms.
5. **Power Spectral Density (PSD):** Frequency-domain PSD estimation (dB) via Fast Fourier Transform (FFT / Periodogram) with noise chromaticity categorization (white vs. pink/brown noise).

---

## 📂 Modular Architecture

```
modelos-tr3/
├── dados/                               # Raw time-series text files (.txt) containing frame bit sizes
├── resultados/                          # Automatically generated plots and numerical metrics
│   ├── passo1_sinais_e_histogramas/     # Discrete time-series plots x(n) and Gaussian-fitted histograms
│   ├── passo2_valores_esperados/        # Consolidated statistical summary report (estatisticas.txt)
│   ├── passo3_facs/                     # Autocorrelation (ACF) and cross-correlation (CCF) curves
│   ├── passo4_descorrelacao/            # Covariance matrices, scatter plots, and decorrelation transforms
│   └── passo6_7_dep_e_espectro/         # Power Spectral Density (PSD in dB) and frequency spectra
├── passo1_2_sinais_e_momentos.py        # Statistical moments and empirical distribution module
├── passo3_facs.py                       # Autocorrelation and cross-correlation module
├── passo4_descorrelacao.py              # Covariance matrix and decorrelation module
├── passo6_7_dep_e_espectro.py           # Spectral analysis and noise classification module
├── executar_tudo.py                     # Automated end-to-end processing pipeline
└── requirements.txt                     # Dependencies (numpy, scipy, matplotlib)
```

---

## 📐 Mathematical Formulation

* **Sample Autocorrelation Function (ACF):**
  $$R_{xx}[k] = \frac{1}{N - k} \sum_{n=0}^{N-k-1} (x[n] - \mu_x)(x[n+k] - \mu_x)$$

* **Power Spectral Density (Wiener–Khinchin Theorem):**
  $$S_{xx}(f) = \mathcal{F}\{ R_{xx}[k] \} = \sum_{k=-\infty}^{\infty} R_{xx}[k] e^{-j 2 \pi f k}$$

---

## 🚀 Execution & Quick Start

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Place raw time-series data files in `dados/`.

3. Execute the automated analysis pipeline:
```bash
python executar_tudo.py
```

All high-resolution figures and statistical tables will be automatically exported to `resultados/`.

---

## 👥 Author
* **João Victor Pariz** ([@VPariz](https://github.com/VPariz)) — Electronics Engineering, Military Institute of Engineering (IME).
