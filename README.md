# Alpha & Factor Analysis of an Indian Large-Cap Stock Portfolio

PCA latent-factor decomposition, CAPM-style alpha/beta estimation against NIFTYBEES (NIFTY 50 proxy), beta-hedging with information-ratio comparison, and a Parkinson-volatility signal test, all on an equal-weight portfolio of 8 NSE stocks.

**Portfolio:** RELIANCE, HINDALCO, SBIN, TCS, TRENT, EICHERMOT, DRREDDY, ITC (all `.NS`)
**Data:** Daily OHLCV from Yahoo Finance via OpenBB (`yfinance` provider)
**Stack:** Python 3.11, pandas, NumPy, scikit-learn, statsmodels, SciPy, seaborn, matplotlib, OpenBB

## Project structure

| Notebook | Purpose |
|---|---|
| `pca_analysis.ipynb` | PCA on daily returns (2023-01 → 2025-12): explained variance, latent factor returns, stock exposures |
| `hedging_portfolio.ipynb` | OLS regression of portfolio returns on NIFTYBEES → alpha, beta; beta-hedged portfolio; information ratio (hedged vs unhedged) |
| `assesing_mkt_volality.ipynb` | Parkinson (high-low) volatility feature, Spearman rank-correlation test against forward returns (2017-01 → 2025-12) |

## Methodology

### 1. PCA latent factors
- Daily simple returns of the 8 stocks (739 observations).
- `sklearn.decomposition.PCA(n_components=3)` on the return covariance matrix.
- Factor returns = centred returns projected on the principal components (`pca.transform`); factor exposures = the component loadings per stock.
- **Result:** the first 3 components explain about **66%** of return variance (PC1 ≈ 37.5%, PC2 ≈ 17%, PC3 ≈ 11%). PC1 loads positively on every stock, so it reads as a market-wide factor. PC2 and PC3 separate stocks by idiosyncratic or sector behaviour.
- **Link to the market:** regressing the factor returns on NIFTYBEES, PC1 has correlation **0.74** (R² ≈ 0.55, β p-value ≈ 0), confirming it is a market factor. PC2 (corr −0.34) and PC3 (corr −0.21) are much weaker.
- Raw-return PCA is dominated by the highest-volatility stock (TRENT) in PC1 and PC2. As a robustness check, PCA on standardised returns gives PC1 ≈ 33.7% explained variance, with PC1 correlation to NIFTYBEES rising to **0.85**.

### 2. Alpha / beta and hedging
- Regress portfolio daily return (y, dependent) on NIFTYBEES daily return (x, independent) with OLS (`statsmodels`): `R_p = α + β·R_m + ε`.
- Hedged portfolio: `R_hedged = R_p − β·R_m` (long portfolio, short β units of NIFTYBEES).
- Compare information ratios (mean / std of daily active return) of the hedged and unhedged portfolio.
- Portfolio is equal-weight (mean of the 8 stock returns), aligned with NIFTYBEES on a common date index. Information ratios are annualised with √252.
- **Result:** β ≈ **1.05** (t ≈ 39.8), R² ≈ 0.68, daily α ≈ 0.0002 (p ≈ 0.30, **not statistically significant**). After hedging, residual β to NIFTYBEES is ≈ 0.
- **Unhedged IR vs NIFTYBEES: 0.69.** **Hedged IR (benchmark-neutral): 0.61.** Standalone Sharpe: portfolio 1.41, NIFTYBEES 1.29, hedged 0.61.
- Interpretation: the 8 stocks only slightly beat the index in-sample, and the alpha is not statistically distinguishable from zero. Hedging removes the market return and leaves a positive, lower-risk-adjusted residual return of about 0.6 Sharpe.

### 3. Parkinson volatility vs forward returns
- Parkinson estimator: `σ = sqrt( 252 · mean( ln(H/L)² / (4·ln 2) ) )` over a 14-day rolling window, then z-scored.
- Spearman rank correlation between the volatility score and the next-day return, pooled over 8 stocks (17,680 observations).
- **Result:** ρ = −0.16%, p = 82.9%.

## How to read the volatility test (hypothesis test)

- H₀: there is no monotonic relationship between Parkinson volatility and next-day return (ρ = 0).
- The rule is to reject H₀ when **p < 0.05** (the significance level), not when p > 0.5.
- Here ρ ≈ 0 and **p = 0.829 > 0.05, so we fail to reject H₀.** There is no statistically significant evidence that Parkinson volatility predicts the *direction* of next-day returns.
- A large p-value means "no evidence of a relationship", not "proof they are unrelated".
- Volatility predicts the *size* of moves, not their sign. The notebook's follow-up test, with absolute next-day return as the target, gives a strongly significant ρ ≈ +0.18 (p ≈ 6e-131), so **reject H₀** there.

## Limitations

- Survivorship and selection bias: the 8 stocks were hand-picked, and several (TRENT, HINDALCO) were strong performers over the sample.
- The 3-year sample (2023–2025) for the regression and PCA is short and covers one market regime.
- Static, in-sample β (no rolling or out-of-sample estimate); the hedge is not rebalanced.
- No transaction costs, borrow/futures costs or slippage for the hedge leg.
- The volatility z-scores use full-sample mean and std (mild look-ahead), and daily observations pooled across stocks are not fully independent.

## Running

```bash
pip install openbb scikit-learn statsmodels scipy seaborn matplotlib pandas numpy
jupyter lab
```

Run the notebooks independently; each downloads its own data (Yahoo Finance values can shift slightly between downloads, so reruns may differ in the last decimals).
