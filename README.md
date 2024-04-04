# Default Probabilities Calibration 

This project aims to enhance structural default risk models by incorporating long-term memory and self-similarity of asset returns. It proposes to investigate two classes of stochastic processes meeting these criteria: fractional Brownian motion and grey noise, or Generalized Grey Brownian Motion (gBM). The goal is to develop emitter-specific calibration algorithms based on historical stock returns to assess corresponding default probabilities.

## Classes

### `Merton`
- **Description:** Implements the Merton model for calculating default probabilities.
- **Usage:** Initialize with company data and call the `calibrate()` method to compute default probabilities.
- **Example:**
  ```python
  model = Merton(ticker, market_cap, debt, T=maturity)
  proba_default = model.calibrate()[2]

### `Necula`
- **Description**: Implements the Necula model for calculating default probabilities.
- **Usage**: Initialize with company data and call the calibrate() method to compute default probabilities.
- **Example**:
    ```python
    model = Necula(ticker, market_cap, debt, T=maturity)
    proba_default = model.calibrate()[1]

### `Rostek`
- **Description**: Implements the Rostek model for calculating default probabilities.
- **Usage**: Initialize with company data and call the calibrate() method to compute default probabilities.
- **Example**:
    ```python
    model = Rostek(ticker, market_cap, debt, T=maturity)
    proba_default = model.calibrate()[1]

### `GreyNoise`
- **Description**: Implements the GreyNoise model for calculating call option prices.
- **Usage**: Initialize with option parameters and call the grey_call_price() method to compute the call option price.
- **Example**:
    ```python
    model = GreyNoise(ticker, market_cap, debt, T=maturity)
    call_price = model.grey_call_price(S0, K, sigma, tau)

### `Tools`
- **Description**: Provides various financial analysis tools and utilities.
- **Usage**: Initialize with relevant data and use the provided methods for computing probabilities, exporting to LaTeX, etc.
- **Example**:
    ```python
    tools = Tools(ticker, market_cap, debt, T=maturity)
    tools.compute_proba_default(maturity=[1, 5, 10], display_graphe=True, display_H_coeff=True, metric='default')

## Functions
'get_data'
- **Description**: Retrieves financial data from an Excel file.
- **Usage**: Provide the file path and optional parameters to specify data retrieval.
- **Example**:
    ```python
    market_cap, debt = get_data(file="Data.xlsx", sheet_name_market_cap="MarketCap", date=['2021-01-01','2021-12-31'], sheet_name_debt="Debt")

## References

[1] Aguilar, J.Ph., Korbel, J. and Coste, C., Series representation of the pricing formula for the European option driven by space-time fractional equation, Fractional Calculus and Applied Analysis 21(4), 981-1004 (2018) 
[2] Della Ratta, L., and Urga, G., Modelling credit spread: a fractional integration approach, Cass Business School working paper (2005) 
[3] Flint, E and Maré, E., Fractional Black-Scholes Option Pricing, Volatility Calibration, and Implied Hurst Exponents, Available at SSRN 2793927 (2016) 
[4] Leccadito, A., Fractional Models to Credit Risk Pricing, PhD thesis university of Bergamo 
[5] Li, K. Q. and Chen, R., Implied Hurst Exponent and Fractional Implied Volatility: A Variance Term Structure Model, Available at SSRN 2383618 (2014) 
[6] Mainardi, F., Luchko, Y. and Pagnini, G., The fundamental solution of the space-time fractional diffusion equation, Fractional Calculus and Applied Analysis 4(2), 153-192 (2001) 
[7] Mainardi, F., Mura, A. and Pagnini, G., The M-Wright function in time-fractional diffusion processes: a tutorial survey, International Journal of Differential Equations (2010) 
[8] Merton, R., On the pricing of corporate debt: the risk structure of interest rates. The Journal of Finance 29(2), 449-470 (1974). 
[9] Necula, C., Option Pricing in a Fractional Brownian Motion Environment. Advances in Economic and Financial Research, DOFIN Working Paper Series, 2, 259-273 (2008) 
[10] Samorodnitsky, G., Long memory and self-similar processes, Annales de la faculté des sciences de Toulouse 15 (1), 107-123 (2006) 
[11] Sliusarentko, O. Y., Vitali, S., Sposinii V., Paradisi, P., Chechkin, A., Castellani, G., Pagnini, G. Finite-energy Lévy-type motion through heterogeneous ensemble of Brownian particles, Journal of Phyics A 52(9), 095601 (2019) 
[12] Vassalou, M. and Xhing, Y., Default risk in equity returns, The Journal of Finance 59(2), 831-868 (2004).
