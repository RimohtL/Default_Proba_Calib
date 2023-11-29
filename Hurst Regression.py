import numpy as np
from scipy import optimize
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


""" Data Importation """

path = "/Users/user/Documents/Cours/CentraleSupelec/Projet/Docs/"
file = path + "Data issuers.xlsx"
market_cap = pd.read_excel(file, sheet_name="Mod Market Cap")
market_cap = market_cap.set_index("Dates").loc['2019-10-28':'2020-10-13']
debt = pd.read_excel(file, sheet_name="Gross Debt", nrows=1)

""" Black-Scholes-Merton Model """


def d1(x, sigma_A, t, T, H, rf, company_debt):
    return ((np.log(x / company_debt)) + rf * (T - t) + 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H)) / (
            sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H))))


def d2(x, sigma_A, t, T, H, rf, company_debt):
    return ((np.log(x / company_debt)) + rf * (T - t) - 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H)) / (
            sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H))))

def d1_hurst(x, sigma_A, t, T ,H, rf, company_debt):
    return ((np.log(x / company_debt)) + rf * (T-t) + (0.5 * sigma_A ** 2) * (T ** (2 * H) - t ** (2 * H))) / (
            sigma_A * (T - t)**H)

def d2_hurst(x, sigma_A, t, T, H, rf, company_debt):
    return d1_hurst(x, sigma_A, t, T, H, rf, company_debt) - sigma_A * (T - t)**H


# inverse the black scholes formula
def merton_formula(x, rf, t, T, H, company_debt, equity_value, sigma_A):
    d1_term = x * norm.cdf(d1(x, sigma_A, t, T, H, rf, company_debt))
    d2_term = company_debt * np.exp(-rf * (T - t)) * norm.cdf(d2(x, sigma_A, t, T, H, rf, company_debt))
    return d1_term - d2_term - equity_value

def update_values_regression_fixed_intercept(Var, delta_t, sigma_A, iteration, plot=True):
    var_tau = np.array(Var)

    # Transformation logarithmique
    log_delta_t = np.log(delta_t)
    log_var_tau = np.log(var_tau)

    fixed_intercept_log_sigma2 = np.log(var_tau[0]) # assuming delta = 1 otherwise H is here

    # Régression linéaire
    X = log_delta_t.reshape(-1, 1)
    y = log_var_tau - fixed_intercept_log_sigma2

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Coefficients de la régression
    slope = model.coef_[0]

    # Calcul de H
    H = slope / 2
    sigma_A_former = sigma_A
    sigma_A = np.sqrt(var_tau[0]) * np.sqrt(int(252/delta_t[0]))

    if plot:
        plt.scatter(log_delta_t, y, label='Données')
        plt.plot(log_delta_t, model.predict(log_delta_t.reshape(-1, 1)), color='red', label='Régression linéaire')
        plt.xlabel('log(Delta t)')
        plt.ylabel('log(Var(tau(Delta t)))')
        plt.title(f"Régression de l'itération {iteration}")
        plt.legend()
        plt.show()

    return sigma_A, sigma_A_former, H

def update_values_regression(Var, delta_t, sigma_A, iteration, plot=True):
    var_tau = np.array(Var)

    # Transformation logarithmique
    log_delta_t = np.log(delta_t)
    log_var_tau = np.log(var_tau)

    # Régression linéaire
    X = log_delta_t.reshape(-1, 1)
    y = log_var_tau

    model = LinearRegression()
    model.fit(X, y)

    # Coefficients de la régression
    intercept = model.intercept_
    slope = model.coef_[0]

    # Calcul de H
    H = slope / 2
    sigma_A_former = sigma_A
    sigma_A = np.sqrt(np.exp(intercept))

    if plot:
        plt.scatter(log_delta_t, y, label='Données')
        plt.plot(log_delta_t, model.predict(X), color='red', label='Régression linéaire')
        plt.xlabel('log(Delta t)')
        plt.ylabel('log(Var(tau(Delta t)))')
        plt.title(f"Régression d l'itération {iteration}")
        plt.legend()
        plt.show()

    return sigma_A, sigma_A_former, H

def BSM_H(ticker, market_cap, debt, T=1, delta=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rf=0, epsilon=10e-3, H0=0.5):
    frequency = []
    for d in delta:
        frequency.append(252 // d)
    company_debt = debt[[ticker]].iloc[0, 0]
    company_market_cap = market_cap[[ticker]].iloc[:, 0]
    sigma_A_former = 0
    H = H0
    sigma_E = np.std(np.diff(np.log(company_market_cap), n=1)) * np.sqrt(frequency[0])
    sigma_A = sigma_E

    n_iter = 1
    while np.abs(sigma_A - sigma_A_former) > epsilon:
        print("Iteration ", n_iter)
        asset_values = {}
        for f in frequency:
            fasset_values = []
            n = company_market_cap.shape[0]
            days = []
            for i in range(n):
                if i % (n // f) == 0:
                    days.append(i)
            for day in days:
                t = day / n
                equity_value = company_market_cap[day]
                # find zero of Merton function, ie asset_value at the current_time
                fasset_values.append(optimize.newton(merton_formula, company_debt,
                                                     args=(rf, t, 1 + T, H, company_debt, equity_value, sigma_A),
                                                     maxiter=100))
            asset_values[f] = fasset_values

        # update values
        Var = []
        for i, f in enumerate(frequency):
            Var.append(np.var(np.diff(np.log(asset_values[f]), n=1)) )# *f)

        n_iter += 1
        print("update values")
        sigma_A, sigma_A_former, H = update_values_regression_fixed_intercept(Var, delta, sigma_A, n_iter, True)
        print(f"sigma= {sigma_A}, H={H}")
    # compute distance to default and default probability
    t = 1
    distance_to_default = d2_hurst(asset_values[frequency[0]][-1], sigma_A, t, t + T, H, rf, company_debt)
    default_probability = (1 - norm.cdf(distance_to_default)) * 100
    return distance_to_default, default_probability

"""
cols = []
for ticker in market_cap.columns:
    if ticker in debt.columns:
        cols.append(ticker)

results = pd.DataFrame(index=["Sigma", "Distance to default", "Default Probability"], columns=cols)

for ticker in cols:
    results[ticker] = BSM(ticker, market_cap, debt)

print(results["CRH LN Equity"])
print(results)
"""

ticker = market_cap.columns[0]
distance_to_default, default_probability = BSM_H(ticker, market_cap, debt)
print(f"{ticker} \nDistance to default {round(distance_to_default, 3)}, Default Probability {round(default_probability, 3)} \n")