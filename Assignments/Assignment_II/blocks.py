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
    A[:] = K + ss.B

    # b. return
    r[:] = rK-par.delta

@nb.njit
def government(par,ini,ss,tau,w,wt,G,LG,L,LY,S,chi,B):

    # Total employment
    LG[:] = G/par.Gamma_G
    L[:] = LG+LY

    # Implied taxation
    tau[:] = (G+w*LG+chi)/(w*(LY+LG))
    wt[:] = (1.0-tau)*w
    # chi[:] = ss.chi

    # dept
    B[:] = 0.0
    # B[:] = (G+w*LG+chi)-(tau*w*L)
    # for t in range(par.T):
        
    #     B_lag = prev(B,t,ini.B)
    #     # tau[t] = ss.tau + par.phi*(B_lag-ss.B)
    #     B[t] = (B_lag + G[t] + chi[t] - tau[t])

    # service flows
    S[:] = np.minimum(G,par.Gamma_G*LG)

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L_hh,L,Y,C_hh,K,I,G,clearing_A,clearing_L,clearing_Y):
    
    # L = L_hh
    # clearing_S[:] = (G+w*LG+chi)-(tau*w*L_hh)
    clearing_A[:] = A-A_hh
    clearing_L[:] = L_hh-L
    I[:] = K-(1.0-par.delta)*lag(ini.K,K)
    clearing_Y[:] = Y-C_hh-I-G