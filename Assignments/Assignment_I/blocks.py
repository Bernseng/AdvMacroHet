import numpy as np
import numba as nb

from GEModelTools import lag, lead

# lags and leads of unknowns and shocks
# K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
# K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

@nb.njit
def production_firm(par,ini,ss,Gamma,K,L0,L1,rK,Y,w0,w1):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*Gamma*K_lag**(par.alpha-1.0)*(L0**((1-par.alpha)/2))*(L1**((1-par.alpha)/2))
    
    # Calculate wages for both labor types
    w0[:] = ((1.0-par.alpha)/2)*Gamma*(K_lag**par.alpha)*L0**((-par.alpha-1)/2)*L1**((1-par.alpha)/2)
    w1[:] = ((1.0-par.alpha)/2)*Gamma*(K_lag**par.alpha)*L1**((-par.alpha-1)/2)*L0**((1-par.alpha)/2)

    # b. production 
    Y[:] = Gamma*(K_lag**par.alpha)*(L0**((1-par.alpha)/2))*(L1**((1-par.alpha)/2))

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rK-par.delta

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L0,L1,L0_hh,L1_hh,Y,C_hh,K,I,clearing_A,clearing_L0,clearing_L1,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L0[:] = L0-L0_hh
    clearing_L1[:] = L1-L1_hh
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I


