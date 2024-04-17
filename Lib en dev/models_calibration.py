import numpy as np
from scipy import optimize
import pandas as pd
from scipy.stats import norm, kurtosis
from sklearn.linear_model import LinearRegression
from scipy.special import gamma
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad
from mpmath import invertlaplace, exp


class CalibrationsModels:
    """
    A class for calibrating financial models.

    Attributes:
        ticker (str): Ticker symbol of the company.
        market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
        debt (pandas.DataFrame): DataFrame containing debt data.
        T (float): Time to maturity (in years).
        frequency (int, optional): Frequency of data (default is 252).
        rf (float, optional): Risk-free interest rate (default is 0).
        epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        company_debt (float): Debt of the company.
        company_market_cap (pandas.Series): Market capitalization of the company.
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        """
        Initialize the CalibrationsModels class with the given parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
            debt (pandas.DataFrame): DataFrame containing debt data.
            T (float): Time to maturity (in years).
            frequency (int, optional): Frequency of data (default is 252).
            rf (float, optional): Risk-free interest rate (default is 0).
            epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        """
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
    """
    A class representing the Merton model for calibrating financial models.

    Inherits from CalibrationsModels.

    Attributes:
        Inherits attributes from CalibrationsModels class.

    Methods:
        d1: Calculate d1 parameter of the Merton model.
        d2: Calculate d2 parameter of the Merton model.
        inversed_formula: Calculate the inverse formula of the Merton model.
        calibrate: Calibrate the Merton model.
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        """
        Initialize the Merton model with the given parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
            debt (pandas.DataFrame): DataFrame containing debt data.
            T (float): Time to maturity (in years).
            frequency (int, optional): Frequency of data (default is 252).
            rf (float, optional): Risk-free interest rate (default is 0).
            epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        """
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)

    def d1(self, x, sigma_A, current_time, mu):
        """
        Calculate the d1 parameter of the Merton model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            current_time (float): Current time.
            mu (float): Mean parameter.

        Returns:
            float: The d1 parameter.
        """
        return ((np.log(x / self.company_debt)) + mu * current_time) / (
                sigma_A * np.sqrt(current_time))

    def d2(self, x, sigma_A, current_time, mu):
        """
        Calculate the d2 parameter of the Merton model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            current_time (float): Current time.
            mu (float): Mean parameter.

        Returns:
            float: The d2 parameter.
        """
        return self.d1(x, sigma_A, current_time, mu) - sigma_A * np.sqrt(current_time)

    def inversed_formula(self, x, current_time, equity_value, sigma_A):
        """
        Calculate the inverse formula of the Merton model.

        Args:
            x (float): Input variable.
            current_time (float): Current time.
            equity_value (float): Equity value.
            sigma_A (float): Volatility parameter.

        Returns:
            float: The inverse formula value.
        """
        mu = self.rf + (sigma_A**2)/2
        d1_term = x * norm.cdf(self.d1(x, sigma_A, current_time, mu))
        d2_term = self.company_debt * np.exp(-self.rf * current_time) * norm.cdf(self.d2(x, sigma_A, current_time, mu))
        return d1_term - d2_term - equity_value

    def calibrate(self):
        """
        Calibrate the Merton model.

        Returns:
            tuple: A tuple containing the calibrated parameters (sigma_A, distance_to_default, default_probability, mu).
        """
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
    """
    A class representing the Necula model for calibrating financial models.

    Inherits from CalibrationsModels.

    Attributes:
        Inherits attributes from CalibrationsModels class.

    Methods:
        d1: Calculate d1 parameter of the Necula model.
        d2: Calculate d2 parameter of the Necula model.
        inversed_formula: Calculate the inverse formula of the Necula model.
        update_values_regression: Update regression values for the Necula model.
        c_H: Calculate c_H parameter of the Necula model.
        sigma_estimate: Estimate sigma parameter of the Necula model.
        calibrate: Calibrate the Necula model.
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), rf=0,
                 epsilon=10e-5):
        """
        Initialize the Necula model with the given parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
            debt (pandas.DataFrame): DataFrame containing debt data.
            T (float): Time to maturity (in years).
            frequency (numpy.ndarray, optional): Array containing frequencies of data (default is 252 divided by [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).
            rf (float, optional): Risk-free interest rate (default is 0).
            epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        """
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5

    def d1(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d1 parameter of the Necula model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Mean parameter.

        Returns:
            float: The d1 parameter.
        """
        return (np.log(x / self.company_debt) + mu * (T - t) + 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
                sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H)))

    def d2(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d2 parameter of the Necula model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Mean parameter.

        Returns:
            float: The d2 parameter.
        """
        return (np.log(x / self.company_debt) + mu * (T - t) - 0.5 * sigma_A ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
                sigma_A * np.sqrt(T ** (2 * H) - t ** (2 * H)))

    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        """
        Calculate the inverse formula of the Necula model.

        Args:
            x (float): Input variable.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            equity_value (float): Equity value.
            sigma_A (float): Volatility parameter.

        Returns:
            float: The inverse formula value.
        """
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = self.company_debt * np.exp(-self.rf * (T - t)) * norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf))
        return d1_term - d2_term - equity_value

    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        """
        Update regression values for the Necula model.

        Args:
            Var (list): List of variance values.
            sigma_A (float): Volatility parameter.
            iteration (int): Iteration number.
            n (int): Number of data points.
            plot (bool, optional): Whether to plot the regression results (default is False).

        Returns:
            tuple: A tuple containing updated sigma_A, previous sigma_A, and H values.
        """
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
        sigma_A = np.exp(intercept / 2) * 252 ** H

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
        """
        Calculate c_H parameter of the Necula model.

        Args:
            t (float): Time.
            delta (float): Delta parameter.
            H (float): Hurst exponent.

        Returns:
            float: The c_H parameter.
        """
        return ((t+delta)**(2*H+1) - t**(2*H+1) - delta**(2*H+1))/((2*H+1)*t*H)

    def sigma_estimate(self, VA, H):
        """
        Estimate sigma parameter of the Necula model.

        Args:
            VA (numpy.ndarray): Array of asset values.
            H (float): Hurst exponent.

        Returns:
            float: The estimated sigma parameter.
        """
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
            sigma_A.append(np.sqrt(var_tau[i]) * ((252 / step[i]) ** H))
        s = np.mean(np.array(sigma_A))

        return s

    def calibrate(self):
        """
        Calibrate the Necula model.

        Returns:
            tuple: A tuple containing calibrated parameters (distance_to_default, default_probability, H, sigma_A, sigma_A_former, mu).
        """
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
        mu = (1/t) * np.log(asset_values[self.frequency[0]][-1]/asset_values[self.frequency[0]][0]) + (sigma_A**2)/2 * (t**(2*H-1))
        # mu = np.mean(np.diff(np.log(asset_values[self.frequency[0]]), n=1)) * self.frequency[0] + self.c_H(t, 1/self.frequency[0], H) * (sigma_A ** 2) / 2
        # mu = ((self.rf + (sigma_A ** 2)) / 2) * (self.c_H(t, 1 / self.frequency[0], H) - 1)
        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100
        return distance_to_default, default_probability, H, sigma_A, sigma_A_former, mu

class Rostek(CalibrationsModels):
    """
    A class representing the Rostek model for calibrating financial models.

    Inherits from CalibrationsModels.

    Attributes:
        Inherits attributes from CalibrationsModels class.

    Methods:
        ro_h: Calculate ro_h parameter of the Rostek model.
        d1: Calculate d1 parameter of the Rostek model.
        d2: Calculate d2 parameter of the Rostek model.
        inversed_formula: Calculate the inverse formula of the Rostek model.
        update_values_regression: Update regression values for the Rostek model.
        c_H: Calculate c_H parameter of the Rostek model.
        calibrate: Calibrate the Rostek model.
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=252 // np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), rf=0,
                 epsilon=10e-5):
        """
        Initialize the Rostek model with the given parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
            debt (pandas.DataFrame): DataFrame containing debt data.
            T (float): Time to maturity (in years).
            frequency (numpy.ndarray, optional): Array containing frequencies of data (default is 252 divided by [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).
            rf (float, optional): Risk-free interest rate (default is 0).
            epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        """
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.H0 = 0.5

    def ro_h(self, H):
        """
        Calculate ro_h parameter of the Rostek model.

        Args:
            H (float): Hurst exponent.

        Returns:
            float: The ro_h parameter.
        """
        if H != 0.5:
            return ((np.sin(pi * (H - 0.5)) / (pi * (H - 0.5))) * ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H))))
        return ((gamma(1.5 - H) ** 2) / (gamma(2 - 2 * H)))

    def d1(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d1 parameter of the Rostek model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Mean parameter.

        Returns:
            float: The d1 parameter.
        """
        roH = self.ro_h(H)
        return (((np.log(x / self.company_debt)) + mu * (T - t) + 0.5 * roH * (sigma_A ** 2) * (
                    (T - t) ** (2 * H))) / (np.sqrt(roH) * sigma_A * ((T - t) ** H)))

    def d2(self, x, sigma_A, t, T, H, mu):
        """
        Calculate the d2 parameter of the Rostek model.

        Args:
            x (float): Input variable.
            sigma_A (float): Volatility parameter.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            mu (float): Mean parameter.

        Returns:
            float: The d2 parameter.
        """
        roH = self.ro_h(H)
        return self.d1(x, sigma_A, t, T, H, mu) - np.sqrt(roH) * sigma_A * ((T - t) ** H)

    def inversed_formula(self, x, t, T, H, equity_value, sigma_A):
        """
        Calculate the inverse formula of the Rostek model.

        Args:
            x (float): Input variable.
            t (float): Current time.
            T (float): Time to maturity.
            H (float): Hurst exponent.
            equity_value (float): Equity value.
            sigma_A (float): Volatility parameter.

        Returns:
            float: The inverse formula value.
        """
        d1_term = x * norm.cdf(self.d1(x, sigma_A, t, T, H, self.rf))
        d2_term = self.company_debt * np.exp(-self.rf * (T - t)) * norm.cdf(self.d2(x, sigma_A, t, T, H, self.rf))
        return d1_term - d2_term - equity_value

    def update_values_regression(self, Var, sigma_A, iteration, n, plot=False):
        """
        Update regression values for the Rostek model.

        Args:
            Var (numpy.ndarray): Array containing variance values.
            sigma_A (float): Volatility parameter.
            iteration (int): Iteration number.
            n (int): Number of data points.
            plot (bool, optional): Whether to plot the results (default is False).

        Returns:
            tuple: A tuple containing updated parameters (sigma_A, sigma_A_former, H).
        """
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
        sigma_A = np.exp(intercept / 2) * 252 ** H

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
        """
        Calculate c_H parameter of the Rostek model.

        Args:
            t (float): Current time.
            delta (float): Delta value.
            H (float): Hurst exponent.

        Returns:
            float: The c_H parameter.
        """
        return ((t+delta)**(2*H+1) - t**(2*H+1) - delta**(2*H+1))/((2*H+1)*t*H)

    def calibrate(self):
        """
        Calibrate the Rostek model.

        Returns:
            tuple: A tuple containing calibrated parameters (distance_to_default, default_probability, H, sigma_A, sigma_A_former, mu).
        """
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


        plt.title(f"Generated timeserie of V_A")
        plt.xlabel("time")
        plt.ylabel("V_A")
        plt.legend()
        plt.show()
        assert len(Mean) == len(self.frequency)
        t = int(n/252)
        mu = (1/t) * np.log(asset_values[self.frequency[0]][-1]/asset_values[self.frequency[0]][0]) + (sigma_A**2)/2 * (t**(2*H-1))
        # mu = np.mean(np.diff(np.log(asset_values[self.frequency[0]]), n=1)) * self.frequency[0] + self.c_H(t, 1 /self.frequency[0], H) * (sigma_A ** 2) / 2
        # mu = ((self.rf + (sigma_A ** 2)) / 2) * (self.c_H(t, 1 / self.frequency[0], H) - 1)
        distance_to_default = self.d2(asset_values[self.frequency[0]][-1], sigma_A, t, t + self.T, H, mu)
        default_probability = (1 - norm.cdf(distance_to_default)) * 100
        return distance_to_default, default_probability, H, sigma_A, sigma_A_former, mu

class GreyNoise(CalibrationsModels):
    """
    A class representing the GreyNoise model for calibrating financial models.

    Inherits from CalibrationsModels.

    Attributes:
        Inherits attributes from CalibrationsModels class.

    Methods:
        grey_call_price: Calculate the Grey call option price.
        inversed_formula: Calculate the inverse formula of the GreyNoise model.
        objective: Calculate the objective function for calibration.
        m_wright: Calculate the M-Wright function.
        Xdensity: Calculate the X-density function.
        CDF: Calculate the cumulative distribution function (CDF).
        equations: Define the equations for solving the calibration problem.
        get_omega: Calculate the omega parameter.
        calibrate: Calibrate the GreyNoise model.
    """

    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        """
        Initialize the GreyNoise model with the given parameters.

        Args:
            ticker (str): Ticker symbol of the company.
            market_cap (pandas.DataFrame): DataFrame containing market capitalization data.
            debt (pandas.DataFrame): DataFrame containing debt data.
            T (float): Time to maturity (in years).
            frequency (int, optional): Frequency of data (default is 252).
            rf (float, optional): Risk-free interest rate (default is 0).
            epsilon (float, optional): Tolerance parameter for convergence (default is 10e-5).
        """
        super().__init__(ticker, market_cap, debt, T, frequency, rf, epsilon)
        self.beta = 0.9

    def grey_call_price(self, S0, K, sigma, tau, Ns=30, Nmu=30):
        """
        Calculate the Grey call option price.

        Args:
            S0 (float): Initial asset price.
            K (float): Strike price.
            sigma (float): Volatility parameter.
            tau (float): Time to maturity (in years).
            Ns (int, optional): Number of iterations for summation (default is 30).
            Nmu (int, optional): Number of iterations for mean calculation (default is 30).

        Returns:
            float: The Grey call option price.
        """
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
        """
        Calculate the inverse formula of the GreyNoise model.

        Args:
            x (float): Input variable.
            K (float): Strike price.
            sigma (float): Volatility parameter.
            tau (float): Time to maturity (in years).
            equity_value (float): Equity value.

        Returns:
            float: The inverse formula value.
        """
        return self.grey_call_price(x, K, sigma, tau) - equity_value

    def objective(self, x, K, sigma, tau, equity_value):
        """
        Calculate the objective function for calibration.

        Args:
            x (float): Input variable.
            K (float): Strike price.
            sigma (float): Volatility parameter.
            tau (float): Time to maturity (in years).
            equity_value (float): Equity value.

        Returns:
            float: The objective function value.
        """
        return abs(self.grey_call_price(x, K, sigma, tau) - equity_value)

    def m_wright(self, nu, x):
        """
        Calculate the M-Wright function.

        Args:
            nu (float): Parameter.
            x (float): Input variable.

        Returns:
            float: The M-Wright function value.
        """
        if x == 0:
            print("x=0")
            return 0
        fp = lambda p: exp(-p ** nu)
        r = (1 / x) ** (1 / nu)

        Mnu = float(invertlaplace(fp, r)) * (r ** (nu + 1)) / nu
        return Mnu

    def Xdensity(self, x, beta, omega, t):
        """
        Calculate the X-density function.

        Args:
            x (float): Input variable.
            beta (float): Beta parameter.
            omega (float): Omega parameter.
            t (float): Time parameter.

        Returns:
            float: The X-density function value.
        """
        d = np.sqrt(-omega) * t ** (beta / 2)
        if d != 0:
            return (0.5 / d) * self.m_wright(beta / 2, abs(x) / d)
        else:
            return None

    def CDF(self, x, beta, omega, t):
        """
        Calculate the cumulative distribution function (CDF).

        Args:
            x (float): Input variable.
            beta (float): Beta parameter.
            omega (float): Omega parameter.
            t (float): Time parameter.

        Returns:
            float: The CDF value.
        """
        return quad(self.Xdensity, -np.inf, x, args=(beta, omega, t))[0]

    def equations(self, p, var, kurt, delta):
        """
        Define the equations for solving the calibration problem.

        Args:
            p (tuple): Tuple containing parameters (sigma, beta).
            var (float): Variance value.
            kurt (float): Kurtosis value.
            delta (float): Delta value.

        Returns:
            tuple: A tuple containing equation values.
        """
        sigma, beta = p
        gamma_3 = gamma(3)
        gamma_5 = gamma(5)

        omega = self.get_omega(sigma, beta)

        eq1 = (-omega * delta ** beta) * (gamma_3 / gamma(beta + 1)) - var
        eq2 = ((-omega * delta ** beta) ** 2) * (gamma_5 / gamma(2 * beta + 1)) - kurt
        return eq1, eq2

    def get_omega(self, sigma_A, beta_A): # order 4 approximation
        """
        Calculate the omega parameter.

        Args:
            sigma_A (float): Volatility parameter.
            beta_A (float): Beta parameter.

        Returns:
            float: The omega parameter.
        """
        if beta_A >= -0.5:
            return - (sigma_A**2) / gamma(1+ 2*beta_A)
        else:
            return 0

    def calibrate(self):
        """
        Calibrate the GreyNoise model.

        Returns:
            tuple: A tuple containing calibrated parameters (distance_to_default, default_probability, beta_A, beta_A_former, sigma_A, sigma_A_former, omega_A).
        """
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

