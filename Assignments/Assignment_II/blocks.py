import numpy as np
import numba as nb

from GEModelTools import prev,next

import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,K,LY,rK,w,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and LY are inputs)
    rK[:] = par.alpha*par.Gamma_Y*(K_lag/LY)**(par.alpha-1.0)

    w[:] = (1.0-par.alpha)*par.Gamma_Y*(K_lag/LY)**par.alpha
    
    # b. production and investment
    Y[:] = par.Gamma_Y*K_lag**(par.alpha)*LY**(1.0-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K 

    # b. return
    r[:] = rK-par.delta

@nb.njit
def government(par,ini,ss,tau,w,wt,G,LG,L,LY,S,chi,B):

    # Total employment
    LG[:] = par.Gamma_G * G
    L[:] = LG + LY

    # Implied taxation
    wt[:] = (1-tau)*w
    tau[:] = (G+wt*LG)/(wt*(LY+LG))

    # dept
    B[:] = -chi

    # service flows
    S[:] = np.minimum(G,LG*par.Gamma_G)


@nb.njit
def market_clearing(par,ini,ss,A,A_hh,LY,LG,L_hh,Y,C_hh,K,I,G,clearing_A,clearing_L,clearing_Y):
    
    # L = L_hh
    clearing_A[:] = A-A_hh
    clearing_L[:] = LY+LG-L_hh
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I-G