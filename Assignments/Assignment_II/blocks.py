import numpy as np
import numba as nb

from GEModelTools import prev,next

import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,K,LY,rK,w,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*par.Gamma_Y*(K_lag/LY)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*par.Gamma_Y*(K_lag/LY)**par.alpha
    
    # b. production and investment
    Y[:] = par.Gamma_Y*K_lag**(par.alpha)*LY**(1-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K + ss.B

    # b. return
    r[:] = rK-par.delta

@nb.njit
def government(par,ini,ss,B,tau,w,wt,G,LG,S,chi):

    tau[:] = ss.tau
    B[:] = ss.B
    G[:] = ss.G
    LG[:] = ss.LG
    S[:] = ss.S
    # S[:] = min(G,LG*par.Gamma_G)
    chi[:] = ss.chi
    # S[:] = ss.S
    wt[:] = (1-tau)*w


@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L,LY,LG,L_hh,Y,C_hh,K,I,G,clearing_A,clearing_L,clearing_Y):
    
    L = L_hh
    clearing_A[:] = A-A_hh
    clearing_L[:] = LY+LG-L
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I-G