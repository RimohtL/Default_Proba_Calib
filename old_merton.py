import pandas as pd
from scipy.stats import norm, kurtosis, skew
from statistics import mean, variance
import scipy.integrate as integrate
from scipy.linalg import solve
from scipy import optimize, special
import numpy as np
import math
import os
import time

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, './Data issuers.xlsx')
dataEquity = pd.read_excel(filename, sheet_name='Mod Market Cap') 
dataEquity = dataEquity.set_index('Dates').loc['2019-10-28':'2020-10-13']
dataDebt = pd.read_excel(filename, sheet_name='Gross Debt').dropna()
#print(dataDebt)
#print(dataEquity)

class calibrationModels():
    def __init__(self, companyEquityTicker = 'CRH LN Equity', timePeriod = 252, horizon = 1,tolerance = 10e-5, riskNeutral = False):
            
        self.companyEquityTicker_       = companyEquityTicker
        self.tolerance_                 = tolerance
        self.timePeriod_                = timePeriod #Correspond au nombre de jours de la période étudiée
        self.horizon_                   = horizon
        self.relativeTime_              = 0 #Correspond au temps relatif où on effectue le calcul
        self.riskFIR_                   = 0 #Risk-free interest rate
        self.riskNeutral_               = riskNeutral

        ### Modèle Merton
        self.sigma_A_                   = 0
        self.sigma_E_                   = 0
        self.sigma_A_history_           = [-100]

        ## Central moments
        self.mean_                      = 0
        self.variance_                  = 0
        self.kurtosis_                  = 0
        self.skewness_                  = 0

        self.asset_values_              = [] #Correspond à V_A
        self.equity_value_              = 0  #Correspond à V_E

        self.nombreIterations_          = 0
        self.companyDebt_               = dataDebt[[self.companyEquityTicker_]].iloc[0,0]
        self.companyEquityListValues_   = dataEquity[[self.companyEquityTicker_]].iloc[:,0]

    def cumulativeGaussianDistribution(self,x):
        return norm.cdf(x)

    def BlackScholesMertonModel(self):

        def d1(x):
            if self.sigma_A_ == 0:
                print('qqqqqqqqqqqqqqq')
            if self.relativeTime_ == 0:
                print('aaaaaaaaaaaaaaa')
            return ( (np.log(x/self.companyDebt_)) + (self.riskFIR_ + 0.5*self.sigma_A_**2)*self.relativeTime_ ) / (self.sigma_A_ * np.sqrt(self.relativeTime_))
            
        def d2(x):
            return d1(x) - self.sigma_A_ * np.sqrt(self.relativeTime_)

        def modelMerton(x):
            leftPart    = x*self.cumulativeGaussianDistribution(d1(x))
            rightPart   = self.companyDebt_*np.exp(-self.riskFIR_ * self.relativeTime_)*self.cumulativeGaussianDistribution(d2(x))
            #rightPart   = self.companyDebt_*1*self.cumulativeGaussianDistribution(d2(x))
            return leftPart - rightPart - self.equity_value_

        #On calcule la première valeur de sigma_E que l'on utilise comme valeur initiale de sigma_A
        self.sigma_E_ = np.std(np.diff(np.log(self.companyEquityListValues_), n = 1))*np.sqrt(self.timePeriod_)
        #On l'ajoute à l'historique des sigma_A
        self.sigma_A_history_.append(self.sigma_E_)
        
        while np.abs(self.sigma_A_history_[-1] - self.sigma_A_history_[-2]) > self.tolerance_:

            self.asset_values_      = []
            self.sigma_A_           = self.sigma_A_history_[-1] #On prend la dernière valeur estimée de sigma_A dans la boucle précédente
            #print(self.companyEquityListValues_)
            for day in range(self.companyEquityListValues_.shape[0]):

                #print(self.horizon_, self.timePeriod_, day)
                print(day,self.relativeTime_)
                self.relativeTime_ = self.horizon_ + (self.timePeriod_ - day - 1)/252
                #rint(self.relativeTime_)
                self.equity_value_ = self.companyEquityListValues_[day]
                self.asset_values_.append(optimize.newton(modelMerton,self.companyDebt_))

            self.sigma_A_history_.append(np.std(np.diff(np.log(self.asset_values_),n=1))*np.sqrt(self.timePeriod_))
            print(f"A l'itération {self.nombreIterations_} sigma = {round(self.sigma_A_history_[-1]*100,2)}% et VA = {self.asset_values_[-1]}")
            self.nombreIterations_  += 1
            
        self.sigma_A_               = self.sigma_A_history_[-1]
        mertonDistanceToDefault     = d2(self.asset_values_[-1])
        mertonDefaultProbability    = (1 - self.cumulativeGaussianDistribution(mertonDistanceToDefault))*100
            
        return self.nombreIterations_, self.asset_values_[-1], self.sigma_A_, mertonDistanceToDefault, mertonDefaultProbability
        

if __name__ == "__main__":

    model = "Merton"
    
    if model == "Merton":
        #####################################################
        ####################### Merton ######################
        #####################################################

        print("\n" + "#"*120 + "\n")
        horizon = input("Entrez l'horizon (en années) auquel on s'intérèsse : ")
        print("\n" + "#"*120 + "\n")   

        valuesMerton = pd.DataFrame({'ticker':[], 'VA': [], 'sigmaA': [], 'distDefault':[], 'probaDefault':[], 'nIter': [], 'execTime': []})
        tickers = ['CRH LN']

        for i in range(len(tickers)):

            ticker = tickers[i]

            print("\n" + "#"*120 + "\n")
            print(f"La boîte à laquelle on s'intérèsse est : {ticker}")
            print("\n" + "#"*120 + "\n")

            #try:    
            start_time = time.time()

            merton = calibrationModels(companyEquityTicker = ticker + ' Equity', horizon = int(horizon))
            nIter, assetValue, sigmaA, distToDefault, defaultProba = merton.BlackScholesMertonModel()
                
            execTime = round(time.time() - start_time, 4)

            print("Le temps d'execution pour Merton est de %s secondes" % execTime)

            if valuesMerton.empty:
                valuesMerton = pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'sigmaA': [sigmaA], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})
            else:
                valuesMerton = pd.concat([valuesMerton,pd.DataFrame({'ticker':[ticker], 'VA':[assetValue], 'sigmaA': [sigmaA], 'distDefault':[distToDefault], 'probaDefault': [defaultProba], 'nIter': [nIter], 'execTime': [execTime]})], ignore_index=True)

            #except:
            #    print(f"Pas de données sur {ticker}")
        print(valuesMerton)