# Financial Models and Tools

This repository contains Python classes and functions for financial modeling and analysis.

## Classes

### `Merton`
- **Description:** Implements the Merton model for calculating default probabilities.
- **Usage:** Initialize with company data and call the `calibrate()` method to compute default probabilities.
- **Example:**
  ```python
  model = Merton(ticker, market_cap, debt, T=maturity)
  proba_default = model.calibrate()[2]