import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models_calibration import Merton, Necula, Rostek


class Tools:
    def __init__(self, ticker, market_cap, debt, T, frequency=252, rf=0, epsilon=10e-5):
        """
        Initialize the Tools class.

        Parameters:
        - ticker: str, the ticker symbol of the company.
        - market_cap: pandas DataFrame, market capitalization data.
        - debt: pandas DataFrame, debt data.
        - T: int, maturity in years.
        - frequency: int, frequency of data.
        - rf: float, risk-free rate.
        - epsilon: float, tolerance for convergence.

        Returns:
        None
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
        


    def compute_proba_default(self, maturity=[1, 2, 5, 10, 15], display_graphe=True, display_H_coeff=True, metric='default'):
        """
        Compute default probabilities using different models and optionally display a graph.

        Parameters:
        - maturity: list of int, maturities in years.
        - display_graphe: bool, whether to display a graph.
        - display_H_coeff: bool, whether to display Hurst coefficients.
        - metric: str, the metric to use ('default', 'sigma', 'mu').

        Returns:
        - proba_merton: array, default probabilities computed by Merton model.
        - proba_necula: array, default probabilities computed by Necula model.
        - proba_rostek: array, default probabilities computed by Rostek model.
        - Hurst_coef: float, Hurst coefficient.
        """

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
            proba_merton[i] = Merton(self.ticker, self.market_cap, self.debt, T=m).calibrate()[index_merton]
            proba_necula[i] = Necula(self.ticker, self.market_cap, self.debt, T=m).calibrate()[index_necula_rostek]
            proba_rostek[i] = Rostek(self.ticker, self.market_cap, self.debt, T=m).calibrate()[index_necula_rostek]
            if display_H_coeff:
                Hurst_coef = Necula(self.ticker, self.market_cap, self.debt, T=m).calibrate()[2]
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
        """
        Export default probabilities to LaTeX format.

        Parameters:
        - liste_ticker: list of str, ticker symbols of the companies.
        - maturities: list of int, maturities in years.
        - metric: str, the metric to use ('default', 'sigma', 'mu').

        Returns:
        None
        """

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

    def default_proba_as_calib(self, ticker, debt, rf=0, T=5, file_path="/Users/user/Documents/Cours/CentraleSupelec/Projet/Docs/Data issuers.xlsx", len_window=252):
        """
        Compute default probabilities using rolling windows and display the results.

        Parameters:
        - ticker: str, the ticker symbol of the company.
        - debt: pandas DataFrame, debt data.
        - rf: float, risk-free rate.
        - T: int, maturity in years.
        - file_path: str, file path for the data.
        - len_window: int, length of each rolling window.

        Returns:
        - proba_merton: array, default probabilities computed by Merton model.
        - proba_rostek: array, default probabilities computed by Rostek model.
        - proba_necula: array, default probabilities computed by Necula model.
        """

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
        """
        Export default probabilities to LaTeX format.

        Parameters:
        - liste_ticker: list of str, ticker symbols of the companies.
        - ratings: list of str, ratings of the companies.
        - maturities: list of int, maturities in years.
        - metric: str, the metric to use ('default', 'sigma', 'mu').

        Returns:
        None
        """
        
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

