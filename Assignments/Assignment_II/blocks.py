import numpy as np
import numba as nb

from GEModelTools import prev,next

import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,K,LY,rK,w,Y):

    K_lag = lag(ini.K,K)

    # LY[:] = ss.LY

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
def government(par,ini,ss,tau,w,wt,G,LG,S,chi,budget):

    tau[:] = ss.tau
    LG[:] = ss.LG
    G[:] = par.Gamma_G*LG
    chi[:] = ss.chi
    # S[:] = ss.S
    S[:] = np.minimum(G,LG*par.Gamma_G)
    # B[:] = tau*w*L_hh-G-LG*w-chi
    wt[:] = (1-tau)*w

    # Government budget constraint
    lhs = G + w * LG + chi
    rhs = tau * w * ss.L_hh
    budget[:] = rhs - lhs



@nb.njit
def market_clearing(par,ini,ss,A,A_hh,LY,LG,L_hh,Y,C_hh,K,I,G,clearing_A,clearing_L,clearing_Y):
    
    # L = L_hh
    clearing_A[:] = A-A_hh
    clearing_L[:] = LY+LG-L_hh
    I[:] = K-(1-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I-G