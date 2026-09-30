
# Probabilistic Forecasting of Solar Cycles via Variational WaveNet Architectures

This repository implements a probabilistic deep learning framework for long-horizon forecasting of monthly sunspot numbers (SSN) and total sunspot area (TSA). Extending deterministic WaveNet + LSTM architectures, this model incorporates a **Variational Recurrent Neural Network (VRNN / VAE)** framework regularized by **KL Divergence** to output explicit uncertainty bounds ($\mu, \sigma^2$).

## Architecture Highlights
* **Feature Extractor:** 1D Dilated Causal Convolutions (WaveNet) with exponential dilation rates ($r \in \{1, 2, 4, \dots, 512\}$) to capture multi-cycle solar dynamo trends without temporal data leakage.
* **Probabilistic Latent Space:** Latent distribution modeling $z \sim \mathcal{N}(\mu_z, \sigma_z^2)$ to capture stochastic solar variability.
* **Loss Function:** Evidence Lower Bound (ELBO) combining Mean Squared Error (MSE) reconstruction loss with a KL Divergence penalty.

## Dataset
* **Source:** WDC-SILSO International Sunspot Number (SSN) v2.0 (Monthly averaged data, 1749–present).
* **Preprocessing:** 13-month moving average smoothing, Min-Max normalization $[0, 1]$, and temporal sequence windowing ($T=528$ months history $\rightarrow N=132$ months forecast).

## Setup & Running

```bash
pip install -r requirements.txt
python evaluate_retrospective.py