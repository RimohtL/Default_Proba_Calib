import numpy as np
from scipy import optimize
import pandas as pd
from scipy.stats import norm

""" Data Importation """

path = "/Users/user/Documents/Cours/CentraleSupelec/Projet/Docs/"
file = path + "Data issuers.xlsx"
market_cap = pd.read_excel(file, sheet_name="Mod Market Cap")
market_cap = market_cap.set_index("Dates").loc['2019-10-28':'2020-10-13']
debt = pd.read_excel(file, sheet_name="Gross Debt", nrows=1)

""" Black-Scholes-Merton Model """

def BSM(ticker, market_cap, debt, T=1, frequency=252, rf=0, epsilon=10e-3):
    company_debt = debt[[ticker]].iloc[0, 0]
    company_market_cap = market_cap[[ticker]].iloc[:, 0]
    current_time = 0
    equity_value = 0
    sigma_A = 0
    sigma_A_former = 0
    asset_values = []

    def d1(x, sigma_A, current_time):
        return ((np.log(x / company_debt)) + (rf + 0.5 * sigma_A ** 2) * current_time) / (
                sigma_A * np.sqrt(current_time))

    def d2(x, sigma_A, current_time):
        return d1(x, sigma_A, current_time) - sigma_A * np.sqrt(current_time)

    # inverse the black scholes formula
    def merton_formula(x, rf, current_time):
        d1_term = x * norm.cdf(d1(x, sigma_A, current_time))
        d2_term = company_debt * np.exp(-rf * current_time) * norm.cdf(d2(x, sigma_A, current_time))
        return d1_term - d2_term - equity_value

    sigma_E = np.std(np.diff(np.log(company_market_cap), n=1)) * np.sqrt(frequency)
    sigma_A = sigma_E

    while np.abs(sigma_A - sigma_A_former) > epsilon:

        asset_values = []

        for dt in range(company_market_cap.shape[0]):
            current_time = T + (frequency - dt - 1) / frequency
            equity_value = company_market_cap[dt]
            # find zero of Merton function, ie asset_value at the current_time
            asset_values.append(optimize.newton(merton_formula, company_debt, args=(rf, current_time)))

        # update of sigma_A and sigma_A_former
        sigma_A_former = sigma_A
        sigma_A = np.std(np.diff(np.log(asset_values), n=1)) * np.sqrt(frequency)

    # compute distance to default and default probability
    distance_to_default = d2(asset_values[-1], sigma_A, current_time)
    default_probability = (1 - norm.cdf(distance_to_default)) * 100

    return distance_to_default, default_probability

""" Test """

for ticker in market_cap.columns:
    if ticker in debt.columns:
        distance_to_default, default_probability = BSM(ticker, market_cap, debt)
        print(f"{ticker} \nDistance to default {round(distance_to_default, 3)}, Default Probability {round(default_probability, 3)} \n")
