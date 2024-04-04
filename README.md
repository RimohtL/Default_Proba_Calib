# Financial Models and Tools

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

