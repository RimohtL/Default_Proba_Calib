import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi
import scipy.stats as si

# %% Methods

def black_scholes_call(S, K, r, sigma, t, T):
    """
    Calcul du prix d'une option d'achat (call) selon le modèle de Black-Scholes.

    Args:
        S (float): Prix actuel de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).

    Returns:
        float: Prix de l'option d'achat (call) selon le modèle de Black-Scholes.
    """

    DT=T-t
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * DT) / (sigma * np.sqrt(DT))
    d2 = d1 - sigma * np.sqrt(DT)
    
    call_option_price = S * si.norm.cdf(d1, 0, 1) - K * np.exp(-r * DT) * si.norm.cdf(d2, 0, 1)
    return call_option_price

def necula_call(S,K,r,sigma,t,T,H):
    """
    Calcul du prix d'une option d'achat (call) selon le modèle de Necula.

    Args:
        S (float): Prix actuel de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        H (float): Paramètre de Hurst.

    Returns:
        float: Prix de l'option d'achat (call) selon le modèle de Necula.
    """

    
    d1=((np.log(S / K)) + r * (T - t) + 0.5 * sigma ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
            sigma * np.sqrt(T ** (2 * H) - t ** (2 * H)))
    d2=((np.log(S / K)) + r * (T - t) - 0.5 * sigma ** 2 * (T ** (2 * H) - t ** (2 * H))) / (
            sigma * np.sqrt(T ** (2 * H) - t ** (2 * H)))
    
    call_option_price = S * si.norm.cdf(d1, 0, 1) - K * np.exp(-r * (T - t)) * si.norm.cdf(d2, 0, 1)
    return(call_option_price)

def ro_h(H):
    """
    Calcul du coefficient ro(H) pour le modèle de Rostek.

    Args:
        H (float): Paramètre de Hurst.

    Returns:
        float: Coefficient ro(H) pour le modèle de Rostek.
    """

    if H!=0.5:
        return( (np.sin(pi*(H-0.5))/(pi*(H-0.5)))*((gamma(1.5-H)**2)/(gamma(2-2*H))) )
    return( (gamma(1.5-H)**2)/(gamma(2-2*H)))

def d1_rostek(S, K, r, sigma, t, T, H, roH):
    """
    Calcul du terme d1 dans la formule de Black-Scholes pour le modèle de Rostek.

    Args:
        S (float): Prix actuel de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        H (float): Paramètre de Hurst.
        roH (float): Coefficient ro(H) pour le modèle de Rostek.

    Returns:
        float: Terme d1 dans la formule de Black-Scholes pour le modèle de Rostek.
    """

    return(
       (np.log(S / K) + r * (T-t) + 0.5* roH * ((sigma )** 2 )* ((T-t)**(2*H)))/(np.sqrt(roH)*sigma*((T-t)**H))
    )

def d2_rostek(S, K, r, sigma, t, T, H, roH):
    """
    Calcul du terme d2 dans la formule de Black-Scholes pour le modèle de Rostek.

    Args:
        S (float): Prix actuel de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        H (float): Paramètre de Hurst.
        roH (float): Coefficient ro(H) pour le modèle de Rostek.

    Returns:
        float: Terme d2 dans la formule de Black-Scholes pour le modèle de Rostek.
    """

    return(
        d1_rostek(S, K, r, sigma, t, T, H, roH) - np.sqrt(roH)*sigma*((T-t)**H)
    )

def rostek_call(S, K, r, sigma, t, T, H):
    """
    Calcul du prix d'une option d'achat (call) selon le modèle de Rostek.

    Args:
        S (float): Prix actuel de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        H (float): Paramètre de Hurst.

    Returns:
        float: Prix de l'option d'achat (call) selon le modèle de Rostek.
    """

    roH=ro_h(H)
    
    d1_term=S * norm.cdf(d1_rostek(S, K, r, sigma, t, T, H, roH))
    d2_term=K * np.exp(-r * (T - t)) * norm.cdf(d2_rostek(S, K, r, sigma, t, T, H, roH))
    return (d1_term - d2_term)

def grey_call_price(beta,S0,K,r,sigma,tau,Ns,Nmu):
    """
    Calcul du prix d'une option d'achat (call) avec mouvement brownien gris

    Args:
        beta (float): Paramètre de Grey.
        S0 (float): Prix actuel de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        tau (float): Temps jusqu'à l'expiration de l'option (en années).
        Ns (int): Nombre d'itérations pour la sommation.
        Nmu (int): Nombre d'itérations pour le calcul de mu.

    Returns:
        float: Prix de l'option d'achat (call) avec mouvement brownien gris
    """


    mu1=-sigma**2/2
    emu=0
    for n in range(Nmu):
        emu+= (-1)**n * gamma(1+2*n)*mu1**n / ( np.math.factorial(n)* gamma(1+2*beta*n)   )
    mu=-np.log(emu)
    x=np.log(S0/K) + r*tau
    c=-mu*tau**beta
    s=0

    for n in range (Ns+1):
        for m in range (1,Ns+1):
            s+= ((-1)**n)   *((-x-mu*tau)**n) * (c**((m-n)/2)) /  (np.math.factorial(n)* gamma(1-beta*((n-m)/2)))

    return(s*K*np.exp(-r*tau)/2 )


# %% Display function

def plot_hurst_influence_europ_call(K = 100, r = 0.05, sigma = 0.2, t = 0, T = 0.25, S0_range = np.linspace(80, 120, 100) , Hl=[0.3, 0.4,0.6,0.7,0.8,0.9]): 
    """
    Trace l'influence de l'exposant de Hurst sur le prix de l'option d'achat européenne.

    Args:
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps jusqu'à l'expiration de l'option (en années).
        S0_range (numpy.ndarray): Plage des prix de l'actif sous-jacent.
        Hl (list): Liste des valeurs de l'exposant de Hurst à tester.

    Returns:
        None
    """


    BS_option_prices=[black_scholes_call(S0, K, r, sigma, t,T) for S0 in S0_range]
    for H in Hl:
        NEC_option_prices=[necula_call(S0, K, r, sigma, t,T, H) for S0 in S0_range]
        plt.plot(S0_range,NEC_option_prices,'--',label='H='+str(H))

    plt.plot(S0_range,BS_option_prices,label='Black Scholes',color='black')
    plt.legend()
    plt.grid()
    plt.title('Influence of the Hurst Exponent on the European Call Price')
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Option Price')

def plot_necula_vs_rostek_europ_call(K = 100, r = 0.05, sigma = 0.2, t = 0, T = 0.25, S0_range = np.linspace(80, 120, 100) , Hl=[0.6, 0.95]):

    """
    Trace les prix des options d'achat européennes selon les formules de Necula et de Rostek.

    Args:
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        t (float): Temps actuel (en années).
        T (float): Temps jusqu'à l'expiration de l'option (en années).
        S0_range (numpy.ndarray): Plage des prix de l'actif sous-jacent.
        Hl (list): duo des valeurs de l'exposant de Hurst à tester pour les formules de Necula et de Rostek.

    Returns:
        None
    """


    H=Hl[0]
    BS_option_prices=[black_scholes_call(S0, K, r, sigma, t,T) for S0 in S0_range]
    ROS_option_prices=[rostek_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]
    NEC_option_prices=[necula_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]

    plt.plot(S0_range,BS_option_prices,label='Black Scholes',color='black')
    plt.plot(S0_range,NEC_option_prices,'--', color='blue',label='Necula H='+str(H))
    plt.plot(S0_range,ROS_option_prices,'--',color='red',label='Rostek H='+str(H))

    H=Hl[1]
    ROS_option_prices=[rostek_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]
    NEC_option_prices=[necula_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]

    plt.plot(S0_range,ROS_option_prices,color='red',label='Rostek H='+str(H))
    plt.plot(S0_range,NEC_option_prices, color='blue',label='Necula H='+str(H))

    plt.legend()
    plt.grid()
    plt.title('Two formulas for the European Call Price')
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Option Price')

def plot_grey_call_price(beta=0.9, S0=150, K=100, r=0.05, sigma=0.25, tau = 0.25, range_k = 30):
    """
    Trace les prix des options d'achat européennes en utilisant la formule de Grey.

    Args:
        beta (float): Paramètre beta de la formule de Grey.
        S0 (float): Prix initial de l'actif sous-jacent.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        tau (float): Temps jusqu'à l'expiration de l'option (en années).
        range_k (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.

    Returns:
        None
    """


    C=[ grey_call_price(beta,S0,K,r,sigma,tau,Ns=k,Nmu=k) for k in range (range_k)]

    plt.plot(C)

def plot_grey_vs_bs_europ_call(beta=0.9, K=100, r=0.05, sigma=0.25, tau = 0.25, t = 0, T = 0.25, S0_range = np.linspace(80, 120, 100), beta1= 0.8, beta2 = 1, Ns=20,Nmu=20): 

    """
    Trace les prix des options d'achat européennes en utilisant les formules de Grey et de Black-Scholes.

    Args:
        beta (float): Paramètre beta pour la formule de Grey.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        tau (float): Temps jusqu'à l'expiration de l'option (en années).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        S0_range (array_like): Plage de valeurs du prix initial de l'actif sous-jacent.
        beta1 (float): Paramètre beta pour la première formule de Grey.
        beta2 (float): Paramètre beta pour la deuxième formule de Grey.
        Ns (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.
        Nmu (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.

    Returns:
        None
    """

    beta=beta1
    Grey_option_prices=[grey_call_price(beta, S0, K, r, sigma, tau,Ns=Ns,Nmu=Nmu) for S0 in S0_range]
    BS_option_prices=[black_scholes_call(S0, K, r, sigma, t,T) for S0 in S0_range]

    plt.plot(S0_range,BS_option_prices,label='Black Scholes',color='black')
    plt.plot(S0_range,Grey_option_prices,color='green',label='Grey beta='+str(beta))

    beta=beta2
    Grey_option_prices=[grey_call_price(beta, S0, K, r, sigma, tau,Ns=Ns,Nmu=Nmu) for S0 in S0_range]

    plt.plot(S0_range,Grey_option_prices,'--',color='green',label='Grey beta='+str(beta))

    plt.legend()
    plt.grid()
    plt.title('Two formulas for the European Call Price')
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Option Price')

def plot_beta_exp_europ_call(beta=0.9, K=100, r=0.05, sigma=0.25, tau = 0.25, t = 0, T = 0.25, S0_range = np.linspace(80, 120, 100),betal=[0.8,0.9,1,1.1, 1.2], Ns=20,Nmu=20):
    """
    Trace les prix des options d'achat européennes en fonction du paramètre beta.

    Args:
        beta (float): Paramètre beta pour la formule de Grey.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        tau (float): Temps jusqu'à l'expiration de l'option (en années).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        S0_range (array_like): Plage de valeurs du prix initial de l'actif sous-jacent.
        betal (list): Liste des valeurs de beta à tracer.
        Ns (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.
        Nmu (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.

    Returns:
        None
    """

    BS_option_prices=[black_scholes_call(S0, K, r, sigma, t,T) for S0 in S0_range]
    for beta in betal:
        Grey_option_prices=[grey_call_price(beta, S0, K, r, sigma, tau,Ns=Ns,Nmu=Nmu) for S0 in S0_range]
        plt.plot(S0_range,Grey_option_prices,'--',label='beta='+str(beta))
    plt.plot(S0_range,BS_option_prices,label='Black Scholes',color='black')
    plt.legend()
    plt.grid()
    plt.title('Influence of the beta Exponent on the European Call Price')
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Option Price')

def plot_necual_rostek_grey_BS_europ_call(beta=0.9, K=100, r=0.05, sigma=0.25, tau = 0.25, t = 0, T = 0.25, S0_range = np.linspace(80, 120, 100),beta1= 0.8, beta2 = 1, Ns=20,Nmu=20, Hl=[0.6, 0.95]): 

    """
    Trace les prix des options d'achat européennes en fonction de différents paramètres.

    Args:
        beta (float): Paramètre beta pour les formules de Grey.
        K (float): Prix d'exercice de l'option.
        r (float): Taux d'intérêt sans risque (annuel).
        sigma (float): Volatilité de l'actif sous-jacent (écart-type).
        tau (float): Temps jusqu'à l'expiration de l'option (en années).
        t (float): Temps actuel (en années).
        T (float): Temps d'expiration de l'option (en années).
        S0_range (array_like): Plage de valeurs du prix initial de l'actif sous-jacent.
        beta1 (float): Valeur du paramètre beta pour la première formule de Grey.
        beta2 (float): Valeur du paramètre beta pour la deuxième formule de Grey.
        Ns (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.
        Nmu (int): Nombre de termes à utiliser pour approximer la somme dans la formule de Grey.
        Hl (list): Liste des valeurs de Hurst à utiliser pour les formules de Rostek et Necula.

    Returns:
        None
    """


    H=Hl[0]
    ROS_option_prices=[rostek_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]
    NEC_option_prices=[necula_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]
    BS_option_prices=[black_scholes_call(S0, K, r, sigma, t,T) for S0 in S0_range]

    plt.plot(S0_range,BS_option_prices,label='Black Scholes',color='black')
    plt.plot(S0_range,NEC_option_prices,'--', color='blue',label='Necula H='+str(H))
    plt.plot(S0_range,ROS_option_prices,'--',color='red',label='Rostek H='+str(H))

    H=Hl[1]
    ROS_option_prices=[rostek_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]
    NEC_option_prices=[necula_call(S0, K, r, sigma, t, T, H) for S0 in S0_range]

    plt.plot(S0_range,ROS_option_prices,color='red',label='Rostek H='+str(H))
    plt.plot(S0_range,NEC_option_prices, color='blue',label='Necula H='+str(H))

    beta=beta1
    Grey_option_prices=[grey_call_price(beta, S0, K, r, sigma, tau,Ns=Ns,Nmu=Nmu) for S0 in S0_range]

    plt.plot(S0_range,Grey_option_prices,color='green',label='Grey beta='+str(beta))

    beta=beta2
    Grey_option_prices=[grey_call_price(beta, S0, K, r, sigma, tau,Ns=Ns,Nmu=Nmu) for S0 in S0_range]

    plt.plot(S0_range,Grey_option_prices,'--',color='green',label='Grey beta='+str(beta))

    plt.legend()
    plt.grid()
    plt.title('Three formulas for the European Call Price')
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Option Price')