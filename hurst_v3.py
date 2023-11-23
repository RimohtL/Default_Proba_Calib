import numpy as np
from scipy import optimize
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit


""" Data Importation """

path = "C:/Users/Amaury/Documents/CentraleSupelec/3A/Projet/"
file = path + "Data issuers.xlsx"
market_cap = pd.read_excel(file, sheet_name="Mod Market Cap")
market_cap = market_cap.set_index("Dates").loc['2019-10-28':'2020-10-13']
debt = pd.read_excel(file, sheet_name="Gross Debt", nrows=1)

""" Black-Scholes-Merton Model """

def d1(x, sigma_A, t, T, H, rf, company_debt):
    return ( (np.log(x / company_debt)) + rf*(T-t) + 0.5 * sigma_A ** 2*(T**(2*H)-t**(2*H) )/ (
    sigma_A*np.sqrt(T**(2*H)-t**(2*H))  ) )

def d2(x, sigma_A, t, T, H,rf,company_debt):
    return ((np.log(x / company_debt)) + rf*(T-t) - 0.5 * sigma_A ** 2*(T**(2*H)-t**(2*H) )/ (
    sigma_A*np.sqrt(T**(2*H)-t**(2*H)) ))

def d2_hurst(x, sigma_A, t, T, H,rf,company_debt):
    return ((np.log(x / company_debt)) + rf*(T-t) - 0.5 * sigma_A ** 2*(T**(2*H)-t**(2*H) )/ (
    sigma_A*np.sqrt((T-t)**(2*H)) ))

# inverse the black scholes formula
def merton_formula(x, rf, t ,T, H,company_debt,equity_value,sigma_A):
    d1_term = x * norm.cdf(d1(x, sigma_A, t,T, H, rf, company_debt))
    d2_term = company_debt * np.exp(-rf * (T-t)) * norm.cdf(d2(x, sigma_A, t,T, H,rf, company_debt))
    return d1_term - d2_term - equity_value

def update_values(Var,frequency,sigma_A):
    H=0.5*np.log(Var[0]/Var[1])/(np.log(frequency[1]/frequency[0]))
    sigma_A_former=sigma_A
    sigma_A=np.sqrt(Var[0])*(frequency[0]**H)
    return(sigma_A,sigma_A_former,H)

def var_relation(f,sigma,H):
    return (np.exp(2* np.log(sigma/(np.exp(H*np.log(f))))))

def update_values2(Var, frequency, sigma_A, H):
    float_frequency=[float(i) for i in frequency]
    sigma_A_former=sigma_A
    popt, pcov = curve_fit(var_relation, float_frequency, Var, p0=np.array([sigma_A,H]))
    sigma_A,H= popt[0], popt[1]

    ydata = np.array(Var)
    y = np.array([var_relation(f, sigma_A, H) for f in frequency])
    residuals = np.array(ydata - y)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R_squared=', r_squared)
    return(sigma_A, sigma_A_former, H)

def get_drift(tau,f,sigma_A,H):
    return( f*(np.mean(tau)+((sigma_A**2)/2)*((1+1/f)**(2*H)-(1/f)**(2*H+1) -1)/(2*H+1)   )    )


def BSM_H(ticker, market_cap, debt, T=1, delta=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rf=0, epsilon=10e-3, H0=0.5):

    frequency = []
    for d in delta:
        frequency.append(252//d)
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
                if i % (n//f) == 0:
                    days.append(i)
            for day in days:
                t = day / n
                equity_value = company_market_cap[day]
                # find zero of Merton function, ie asset_value at the current_time
                fasset_values.append(optimize.newton(merton_formula, company_debt, args=(rf, t, 1+T, H, company_debt, equity_value, sigma_A), maxiter=100))
            asset_values[f] = fasset_values


        # update values
        Var = []
        for f in frequency:
            Var.append(np.var(np.diff(np.log(asset_values[f]), n=1)))

        sigma_A, sigma_A_former, H = update_values2(delta, asset_values, sigma_A, H)
        print(f"sigma= {sigma_A} %, H={H}")
        n_iter += 1
    mu=get_drift(asset_values[frequency[0]],frequency[0],sigma_A,H)
    print('Drift=', mu)
    # compute distance to default and default probability
    distance_to_default = - d2_hurst(company_market_cap.iloc[-1], sigma_A, 1, 1+T, H, rf, company_debt)
    default_probability = (1 - norm.cdf(distance_to_default)) * 100
    return distance_to_default, default_probability


# REMINDER : T = Debt Maturity => so if we want T = 1 then T = Trading period + 1
# Trading period = 252 days here

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

ticker = market_cap.columns[1]
distance_to_default, default_probability = BSM_H(ticker, market_cap, debt)
print(f"{ticker} \nDistance to default {round(distance_to_default, 3)}, Default Probability {round(default_probability, 3)} \n")