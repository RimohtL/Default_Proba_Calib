import numpy as np
from scipy import optimize
import pandas as pd
import scipy.stats as si
from scipy.stats import norm, kurtosis
from sklearn.linear_model import LinearRegression
from scipy.special import gamma
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
from mpmath import invertlaplace, exp


class CalibrationsModels:
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        self.ticker = ticker
        self.market_cap = market_cap
        self.debt = debt
        self.T = T
        self.frequency = frequency
        self.rf = rf
        self.epsilon = epsilon
        self.company_debt = debt[[ticker]].iloc[0, 0]
        self.company_market_cap = market_cap[[ticker]].iloc[:, 0]


class Merton(CalibrationsModels):
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)

    def d1(self, x, sigma_A, current_time, mu):
        return ((np.log(x / self.company_debt)) + mu * current_time) / (
                sigma_A * np.sqrt(current_time))

    def d2(self, x, sigma_A, current_time, mu):
        return self.d1(x, sigma_A, current_time, mu) - sigma_A * np.sqrt(current_time)

    def inversed_formula(self, x, current_time, equity_value, sigma_A):
        mu = rf + (sigma_A**2)/2
        d1_term = x * norm.cdf(self.d1(x, sigma_A, current_time, mu))
        d2_term = self.company_debt * np.exp(-self.rf * current_time) * norm.cdf(self.d2(x, sigma_A, current_time, mu))
        return d1_term - d2_term - equity_value

    def calibrate(self):
        current_time = 0
        sigma_A_former = 0
        asset_values = []

        sigma_E = np.std(np.diff(np.log(self.company_market_cap), n=1)) * np.sqrt(self.frequency)
        sigma_A = sigma_E

        while np.abs(sigma_A - sigma_A_former) > self.epsilon:

            asset_values = []

            for dt in range(self.company_market_cap.shape[0]):
                current_time = self.T + (self.company_market_cap.shape[0] - dt - 1) / self.frequency
                equity_value = self.company_market_cap[dt]
                # find zero of Merton function, ie asset_value at the current_time
                asset_values.append(optimize.newton(self.inversed_formula, self.company_debt,
                                                    args=(current_time, equity_value, sigma_A), maxiter=50))

            # update of sigma_A and sigma_A_former
            sigma_A_former = sigma_A
            sigma_A = np.std(np.diff(np.log(asset_values), n=1)) * np.sqrt(self.frequency)

        mu = np.mean(np.diff(np.log(asset_values), n=1)) * self.frequency + (sigma_A ** 2) / 2
        # compute distance to default and default probability
        distance_to_default = self.d2(asset_values[-1], sigma_A, current_time, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100

        # CHECK AND VERIFY MU VALUES FOR EVERY MODELS => CHANGE OF SIGN
        return sigma_A, distance_to_default, default_probability, mu


class Necula(CalibrationsModels):
    def __init__(self, ticker, market_cap, debt, T, frequency=252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), rf=0,
                 epsilon=10e-5):
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5

    # d1 and d2 from Necula
    def d1(self, x, sigma_A, t, T, H, mu):
        return (np.log(x / self.company_debt) + mu * (T - t) + 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
                sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H)))

    def d2(self, x, sigma_A, t, T, H, mu):
        return (np.log(x / self.company_debt) + mu * (T - t) - 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
                sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H)))

    # inverse the black scholes formula with Necula's expresions for d1 and d2
    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = self.company_debt * np.exp(-self.rf * (T - t)) * norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf))
        return d1_term - d2_term - equity_value

    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        var_tau = np.array(Var)

        delta_t = n // self.frequency

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
        sigma_A = np.exp(intercept / 2) * n ** H

        if plot:
            plt.scatter(log_delta_t, y, label='Données')
            plt.plot(log_delta_t, model.predict(X), color='red', label='Régression linéaire')
            plt.xlabel('log(Delta t)')
            plt.ylabel('log(Var(tau(Delta t)))')
            plt.title(f"Régression d l'itération {iteration}")
            plt.legend()
            plt.show()

        return sigma_A, sigma_A_former, H

    def c_H(self, t, delta, H):
        return ((t+delta)**(2*H+1) - t**(2*H+1) - delta**(2*H+1))/((2*H+1)*t*H)

    def sigma_estimate(self, VA, H):
        VA = np.array(VA)
        n = VA.shape[0]
        step = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # delta_t=step/n

        # Variance of log returns
        Var = []
        for s in step:
            asset_value_resampled = np.array([VA[i] for i in range(0, n, s)])
            log_ret = np.diff(np.log(asset_value_resampled))
            Var.append(np.var(log_ret))
        var_tau = np.array(Var)

        sigma_A = []
        for i in range(len(step)):
            sigma_A.append(np.sqrt(var_tau[i]) * ((n / step[i]) ** H))
        s = np.mean(np.array(sigma_A))

        return s

    def calibrate(self):
        sigma_A_former = 0
        H = self.H0
        H_former = 0
        sigma_E = np.std(np.diff(np.log(self.company_market_cap), n=1)) * np.sqrt(self.frequency[0])
        sigma_A = sigma_E

        n_iter = 1
        while np.abs(sigma_A - sigma_A_former) > self.epsilon or np.abs(H - H_former) > self.epsilon:
            asset_values = {}
            for f in self.frequency:
                fasset_values = []
                n = self.company_market_cap.shape[0]
                days = []
                for i in range(n):
                    if i % (n // f) == 0:
                        days.append(i)
                for day in days:
                    t = day / n
                    equity_value = self.company_market_cap[day]
                    # find zero of Merton function, ie asset_value at the current_time
                    fasset_values.append(optimize.newton(self.inversed_formula, self.company_debt,
                                                         args=(t, t + self.T, H, equity_value, sigma_A), # 1 + self.T
                                                         maxiter=100))
                asset_values[f] = fasset_values

            # update values
            Var = []
            for i, f in enumerate(self.frequency):
                Var.append(np.var(np.diff(np.log(asset_values[f]), n=1)))

            Mean = []
            for i, f in enumerate(self.frequency):
                Mean.append(np.mean(np.diff(np.log(asset_values[f]), n=1)))

            n_iter += 1
            H_former = H
            sigma_A, sigma_A_former, H = self.update_values_regression(Var, sigma_A, n_iter, n, False)
            # sigma_A = self.sigma_estimate(asset_values[self.frequency[0]], H)

        assert len(Mean) == len(self.frequency)
        t = int(n/252)
        mu = np.mean(np.diff(np.log(asset_values[self.frequency[0]]), n=1)) * self.frequency[0] + self.c_H(t, 1/self.frequency[0], H) * (sigma_A ** 2) / 2
        # mu = ((self.rf + (sigma_A ** 2)) / 2) * (self.c_H(t, 1 / self.frequency[0], H) - 1)
        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100
        return distance_to_default, default_probability, H, sigma_A, sigma_A_former, mu


class Rostek(CalibrationsModels):
    def __init__(self, ticker, market_cap, debt, T, frequency=252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), rf=0,
                 epsilon=10e-5):
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5

    # d1 and d2 from Rostek
    def ro_h(self, H):
        if H != 0.5:
            return ((np.sin(pi * (H - 0.5)) / (pi * (H - 0.5))) * ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H))))
        return ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H)))

    def d1(self, x, sigma_A, t, T, H, mu):
        roH = self.ro_h(H)
        return (((np.log(x / self.company_debt)) + mu * (T - t) + 0.5 * roH * (sigma_A ** 2) * (
                    (T - t) ** (2 * H))) / (np.sqrt(roH) * sigma_A * ((T - t) ** H)))

    def d2(self, x, sigma_A, t, T, H, mu):
        roH = self.ro_h(H)
        return self.d1(x, sigma_A, t, T, H, mu) - np.sqrt(roH) * sigma_A * ((T - t) ** H)

    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = self.company_debt * np.exp(-self.rf * (T - t)) * norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf))
        return d1_term - d2_term - equity_value

    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        var_tau = np.array(Var)

        delta_t = n // self.frequency

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
        sigma_A = np.exp(intercept / 2) * n ** H

        if plot:
            plt.scatter(log_delta_t, y, label='Données')
            plt.plot(log_delta_t, model.predict(X), color='red', label='Régression linéaire')
            plt.xlabel('log(Delta t)')
            plt.ylabel('log(Var(tau(Delta t)))')
            plt.title(f"Régression d l'itération {iteration}")
            plt.legend()
            plt.show()

        return sigma_A, sigma_A_former, H

    def c_H(self, t, delta, H):
        return ((t+delta)**(2*H+1) - t**(2*H+1) - delta**(2*H+1))/((2*H+1)*t*H)

    def calibrate(self):
        sigma_A_former = 0
        H = self.H0
        H_former = 0
        sigma_E = np.std(np.diff(np.log(self.company_market_cap), n=1)) * np.sqrt(self.frequency[0])
        sigma_A = sigma_E

        n_iter = 1
        plt.figure()
        while np.abs(sigma_A - sigma_A_former) > self.epsilon or np.abs(H - H_former) > self.epsilon:
            asset_values = {}
            for f in self.frequency:
                fasset_values = []
                n = self.company_market_cap.shape[0]
                days = []
                for i in range(n):
                    if i % (n // f) == 0:
                        days.append(i)
                for day in days:
                    t = day / n
                    equity_value = self.company_market_cap[day]
                    # find zero of Merton function, ie asset_value at the current_time
                    fasset_values.append(optimize.newton(self.inversed_formula, self.company_debt,
                                                         args=(t, t + self.T, H, equity_value, sigma_A),
                                                         maxiter=100))
                asset_values[f] = fasset_values
            plt.plot(asset_values[self.frequency[0]], label=f"itération {n_iter}")

            # update values
            Var = []
            for i, f in enumerate(self.frequency):
                Var.append(np.var(np.diff(np.log(asset_values[f]), n=1)))

            Mean = []
            for i, f in enumerate(self.frequency):
                Mean.append(np.mean(np.diff(np.log(asset_values[f]), n=1)))

            n_iter += 1
            H_former = H
            sigma_A, sigma_A_former, H = self.update_values_regression(Var, sigma_A, n_iter, n, False)


        plt.title(f"Generated timeserie of V_A {ticker}")
        plt.xlabel("time")
        plt.ylabel("V_A")
        plt.legend()
        plt.show()
        assert len(Mean) == len(self.frequency)
        t = int(n/252)
        mu = np.mean(np.diff(np.log(asset_values[self.frequency[0]]), n=1)) * self.frequency[0] + self.c_H(t, 1 /self.frequency[0], H) * (sigma_A ** 2) / 2
        # mu = ((self.rf + (sigma_A ** 2)) / 2) * (self.c_H(t, 1 / self.frequency[0], H) - 1)
        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100
        return distance_to_default, default_probability, H, sigma_A, sigma_A_former, mu


class GreyNoise(CalibrationsModels):
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.beta = 0.9

    def grey_call_price(self, S0, K, sigma, tau, Ns=30, Nmu=30):
        if S0 / K <= 0:
            return 0

        mu1 = -sigma ** 2 / 2

        emu = 0
        for n in range(Nmu):
            emu += (-1) ** n * gamma(1 + 2 * n) * mu1 ** n / (np.math.factorial(n) * gamma(1 + 2 * self.beta * n))
        mu = -np.log(emu)

        x = np.log(S0 / K) + self.rf * tau

        c = -mu * tau ** self.beta
        s = 0

        for n in range(Ns+1):
            for m in range(1, Ns+1):
                s += ((-1) ** n) * ((-x - mu * tau) ** n) * (c ** ((m - n) / 2)) / (
                            np.math.factorial(n) * gamma(1 - self.beta * ((n - m) / 2)))

        return s * K * np.exp(-self.rf * tau) / 2

    def inversed_formula(self, x, K, sigma, tau, equity_value):
        return self.grey_call_price(x, K, sigma, tau) - equity_value

    def objective(self, x, K, sigma, tau, equity_value):
        return abs(self.grey_call_price(x, K, sigma, tau) - equity_value)

    def m_wright(self, nu, x):
        if x == 0:
            print("x=0")
            return 0
        fp = lambda p: exp(-p ** nu)
        r = (1 / x) ** (1 / nu)

        Mnu = float(invertlaplace(fp, r)) * (r ** (nu + 1)) / nu
        return Mnu

    def Xdensity(self, x, beta, omega, t):
        d = np.sqrt(-omega) * t ** (beta / 2)
        if d != 0:
            return (0.5 / d) * self.m_wright(beta / 2, abs(x) / d)
        else:
            return None

    def CDF(self, x, beta, omega, t):
        return quad(self.Xdensity, -np.inf, x, args=(beta, omega, t))[0]

    def equations(self, p, var, kurt, delta):
        sigma, beta = p
        gamma_3 = gamma(3)
        gamma_5 = gamma(5)

        omega = self.get_omega(sigma, beta)

        eq1 = (-omega * delta ** beta) * (gamma_3 / gamma(beta + 1)) - var
        eq2 = ((-omega * delta ** beta) ** 2) * (gamma_5 / gamma(2 * beta + 1)) - kurt
        return eq1, eq2

    def get_omega(self, sigma_A, beta_A): # order 4 approximation
        if beta_A >= -0.5:
            return - (sigma_A**2) / gamma(1+ 2*beta_A)
        else:
            return 0

    def calibrate(self):
        omega_A = 0
        beta_A_former = 0
        sigma_A_former = 0
        sigma_A = 0.2
        beta_A = 0.5
        initial_guesses = (sigma_A, beta_A)

        while np.abs(sigma_A - sigma_A_former) > self.epsilon and np.abs(beta_A - beta_A_former) > self.epsilon:
            asset_values = []

            for dt in range(self.company_market_cap.shape[0]):
                current_time = self.T + (self.company_market_cap.shape[0] - dt - 1) / self.frequency
                equity_value = self.company_market_cap[dt]
                # STUDY THE PROBLEM OF CONVERGENCE OF NEWTON FOR A GIVEN DT
                try:
                    asset_values.append(optimize.newton(self.inversed_formula, self.company_debt, args=(self.company_debt, sigma_A, current_time, equity_value), maxiter=50))

                #asset_values.append(optimize.minimize(self.objective, x0=self.company_debt, method='BFGS', args=(self.company_debt, sigma_A, current_time, equity_value), options={'maxiter': 50}).x[0])
                    print(asset_values[-1])
                except:
                    pass
                # TRY TO INVERT INTEGRAL CALL FORMULA INSTEAD OF SUM FORM

            # update values
            print("asset values", asset_values)
            var_ = np.var(np.diff(np.log(asset_values), n=1))
            kurt_ = kurtosis(np.diff(np.log(asset_values), n=1))
            sigma_A_former = sigma_A
            beta_A_former = beta_A
            result = fsolve(self.equations, initial_guesses, args=(var_, kurt_, self.T/len(asset_values)))
            print(result)
            sigma_A, beta_A = result[0], result[1]
            initial_guesses = (max(sigma_A, 1e-7), max(beta_A, 1e-7))
            self.beta = beta_A

            omega_A = self.get_omega(sigma_A, beta_A)

        distance_to_default = np.log(asset_values[-1]/self.company_debt) + (self.rf + omega_A) * self.T
        default_probability = (1 - self.CDF(distance_to_default, beta_A, omega_A, self.T)) * 100
        return distance_to_default, default_probability, beta_A, beta_A_former, sigma_A, sigma_A_former, omega_A



path = "/Users/user/Documents/Cours/CentraleSupelec/Projet/Docs/"
file = path + "Data issuers.xlsx"
market_cap = pd.read_excel(file, sheet_name="Mod Market Cap")
market_cap = market_cap.set_index("Dates").loc['2019-10-28':'2020-10-13']
#market_cap = market_cap.set_index("Dates").loc['2017-10-13':'2018-10-01']
#market_cap = market_cap.set_index("Dates").loc['2010-10-14':'2020-10-13']
print(market_cap)
debt = pd.read_excel(file, sheet_name="Gross Debt", nrows=1)

#ticker = "CRH LN Equity"
ticker = "MRK GY Equity"

print(debt.columns)

rf = 0.
T = 5
"""
model = Merton(ticker, market_cap, debt, T=T, rf=rf)

print(model.calibrate())

model = Necula(ticker, market_cap, debt, T=T, rf=rf)

print(model.calibrate())

model = Rostek(ticker, market_cap, debt, T=T, rf=rf)

print(model.calibrate())
"""
# model = GreyNoise(ticker, market_cap, debt, T=1)

# print(model.calibrate())

class Tools:
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        self.ticker = ticker
        self.market_cap = market_cap
        self.debt = debt
        self.T = T
        self.frequency = frequency
        self.rf = rf
        self.epsilon = epsilon
        self.company_debt = debt[[ticker]].iloc[0, 0]
        self.company_market_cap = market_cap[[ticker]].iloc[:, 0]


    def compute_proba_default(self, maturity=[1, 2, 5, 10, 15], display_graphe=True, display_H_coeff=True, metric='default'):

        proba_merton = np.zeros(len(maturity))
        proba_necula = np.zeros(len(maturity))
        proba_rostek = np.zeros(len(maturity))

        if metric == 'default':
            index_merton = 2
            index_necula_rostek = 1
        elif metric == 'sigma':
            index_merton = 0
            index_necula_rostek = 3
        for i, m in enumerate(maturity):
            proba_merton[i] = Merton(self.ticker, self.market_cap, debt, T=m).calibrate()[index_merton]
            proba_necula[i] = Necula(self.ticker, self.market_cap, debt, T=m).calibrate()[index_necula_rostek]
            proba_rostek[i] = Rostek(self.ticker, self.market_cap, debt, T=m).calibrate()[index_necula_rostek]
            if display_H_coeff:
                Hurst_coef = Necula(self.ticker, self.market_cap, debt, T=m).calibrate()[2]
                print(f"{self.ticker} Hurst maturité {m}: {Hurst_coef}")

        if display_graphe:
            plt.figure()
            plt.plot(maturity, proba_merton, label="Merton")
            plt.plot(maturity, proba_necula, label="Necula")
            plt.plot(maturity, proba_rostek, label="Rostek")
            plt.legend()
            plt.title(f"Default probability as a function of maturity T for {self.ticker}")
            plt.xlabel("T")
            plt.ylabel("Probability")
            plt.show()

        return proba_merton, proba_necula, proba_rostek, Hurst_coef

    def export_latex(self, liste_ticker, maturities=[1, 5, 10], metric='default'):
        for maturity in maturities:
            print(f"Maturity : {maturity}")
            export = ''
            if metric == 'default':
                index_merton = 2
                index_necula_rostek = 1
            elif metric == 'sigma':
                index_merton = 0
                index_necula_rostek = 3
            elif metric == 'mu':
                index_merton = 3
                index_necula_rostek = 5
            unit=""
            if metric == "default":
                unit = "%"
            for ticker_ in liste_ticker:
                if ticker_ == "CGG FP Equity":
                    break
                export += str(ticker_)[:-7]
                export += ' & '
                proba_merton = np.round(Merton(ticker_, self.market_cap, self.debt, T=maturity).calibrate()[index_merton],2)
                export += str(proba_merton) + unit + ' & '
                proba_necula = np.round(Necula(ticker_, self.market_cap, self.debt, T=maturity).calibrate()[index_necula_rostek], 2)
                export += str(proba_necula) + unit + ' & '
                proba_rostek = np.round(Rostek(ticker_, self.market_cap, self.debt, T=maturity).calibrate()[index_necula_rostek], 2)
                export += str(proba_rostek) + unit
                if metric == 'default':
                    export += ' & '
                    Hurst_coef = np.round(Necula(ticker_, self.market_cap, self.debt, T=maturity).calibrate()[2], 3)
                    export += str(Hurst_coef)
                export += ' \\\ \n'
            print(export)

    def default_proba_as_calib(self, ticker, debt, rf=0, T=5, file_path=file, len_window=252):
        market_cap = pd.read_excel(file_path, sheet_name="Mod Market Cap")
        market_cap = market_cap.set_index("Dates")
        n = market_cap.shape[0]
        n_window = int(n/len_window)
        proba_merton = np.zeros(n_window)
        proba_rostek = np.zeros(n_window)
        proba_necula = np.zeros(n_window)
        for i in range(0, n_window):
            market_cap_window = market_cap.iloc[i*len_window:(i+1)*len_window]
            print(market_cap_window)
            model = Merton(ticker, market_cap_window, debt, T=T, rf=rf)
            proba_merton[i] = model.calibrate()[2]
            model = Necula(ticker, market_cap_window, debt, T=T, rf=rf)
            proba_necula[i] = model.calibrate()[1]
            model = Rostek(ticker, market_cap_window, debt, T=T, rf=rf)
            proba_rostek[i] = model.calibrate()[1]

        plt.figure()
        plt.plot(range(0, n_window), proba_merton, label="Merton")
        plt.plot(range(0, n_window), proba_rostek, label="Rostek")
        plt.plot(range(0, n_window), proba_necula, label="Necula")
        plt.legend()
        plt.xlabel("N_window")
        plt.ylabel("Proba")
        plt.title(f"Proba de défaut à maturité {T} ans pour {ticker}")
        plt.show()
        return proba_merton, proba_rostek, proba_necula

    def export_latex_2(self, liste_ticker, ratings=None, maturities=[1, 5, 10], metric='default'):
        models = (Merton, Necula, Rostek)
        index = [2, 1, 1]
        if metric == 'sigma':
            index = [0, 3, 3]
        elif metric == 'mu':
            index = [3, 5, 5]
        elif metric == 'H':
            index = [2, 2, 2]
        for j, ticker_ in enumerate(liste_ticker):
            if ticker_ == "CGG FP Equity":
                break
            export = str(ticker_)[:-7]
            if ratings is not None:
                export += " & " + ratings[j]
            for i, model in enumerate(models):
                for maturity in maturities:
                    unit = ""
                    if metric == "default":
                        unit = "\%"
                    value = np.round(
                        model(ticker_, self.market_cap, self.debt, T=maturity).calibrate()[index[i]], 2)
                    export += " & " + str(value) + unit
            export += ' \\\ '
            print(export)


tools = Tools(ticker, market_cap, debt, T=1)

#tools.default_proba_as_calib(ticker, debt, T=5)

tools.compute_proba_default()

#'SU FP Equity' bug

liste_equity = ['SAP GY Equity', 'MRK GY Equity', 'AI FP Equity', 'CRH LN Equity',
                'SRG IM Equity', 'DAI GY Equity', 'VIE FP Equity', 'AMP IM Equity',
                'FR FP Equity', 'EO FP Equity', 'GET FP Equity', 'LHA GY Equity',
                'PIA IM Equity', 'CO FP Equity']

ratings = ['A', 'A', 'A-', 'BBB+', 'BBB+', 'BBB+', 'BBB', 'BB+', 'BB+', 'BB', 'BB-',
           'BB-', 'B+', 'B']
#liste_equity = debt.columns
#liste_equity = debt.columns[0:3]

#tools.export_latex(liste_equity, metric="default", maturities=[1, 5, 10])
#tools.export_latex_2(liste_equity, ratings,  metric="default", maturities=[1, 5, 10])
#print("\n")
#tools.export_latex_2(liste_equity,  metric="mu", maturities=[1, 5, 10])
#print("\n")
#tools.export_latex_2(liste_equity,  metric="sigma", maturities=[1, 5, 10])
#print("\n")
#tools.export_latex_2(liste_equity,  metric="H", maturities=[1, 5, 10])

#tools.export_latex(liste_equity, metric="mu")

#tools.export_latex(liste_equity, metric="sigma")

