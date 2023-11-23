import time
import numpy as np
from scipy import optimize
from scipy.optimize import minimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    

    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):

        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    for i_fix in range(par.Nfix):
        
        # a. raw value
        ell = 1.0
        y = ss.wt*ell*par.z_grid
        c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
        v_a = (1+ss.r)*c**(-par.sigma)

        # b. expectation
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a

def government_utility(x, model):

    par = model.par
    ss = model.ss

    # Update model with new tau
    tau = x

    # Calculate government variables based on tau
      # Calculate government services

    # Update the households' decisions based on the new government variables
    model.solve_hh_ss()
    model.simulate_hh_ss()

    # Calculate and return the negative of the expected discounted utility
    return -np.sum(ss.du, axis=1).sum()

def obj_ss(x,model,do_print=False):

    KL = x[0]
    # tau = x[1] if gov else 0.0

    par = model.par
    ss = model.ss

    # a. firms
    ss.rK = par.alpha*par.Gamma_Y*(KL)**(par.alpha-1)
    ss.w = (1.0-par.alpha)*par.Gamma_Y*(KL)**par.alpha

    # b. arbitrage
    ss.r = ss.rK - par.delta
    
    # c. government
    # ss.LG = LG
    # ss.tau = tau
    ss.G = par.Gamma_G*ss.LG
    ss.LG = ss.tau*ss.L_hh - (ss.G + ss.chi)/ss.w
    ss.S = min(ss.G, par.Gamma_G*ss.LG)

    # d. households
    ss.wt = (1-ss.tau)*ss.w
    
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # e. market clearing
    ss.B = 0.0
    ss.L = ss.L_hh
    ss.LY = ss.L - ss.LG
    ss.K = KL*ss.LY
    ss.Y = par.Gamma_Y*ss.K**(par.alpha)*ss.LY**(1-par.alpha)
    ss.I = par.delta*ss.K
    ss.A = ss.K + ss.B
    ss.C_hh = ss.Y - ss.I - ss.G
    ss.clearing_A = ss.A - ss.A_hh
    ss.clearing_L = ss.LY + ss.LG - ss.L
    ss.clearing_Y = ss.Y - (ss.C_hh+ss.I+ss.G)

    return ss.clearing_A


def find_ss(model,tau,do_print=False):
    """ find the steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    # tau_guess = 0.1
    # LG_guess = 0.1
    # if do_print: 
    #     print(f'starting at tau={tau_guess:.4f}')

    #  Government
    # ss.LG = LG
    # ss.chi = par.chi_ss
    ss.tau = tau
    # ss.LG = par.LG_ss
    # ss.LG = ss.tau*ss.L_hh - (ss.G + ss.chi)/ss.w

    KL_min = ((1/par.beta+par.delta-1)/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1)) + 1e-2
    KL_max = (par.delta/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1))-1e-2
    KL_mid = (KL_min+KL_max)/2 # middle point between max values as initial capital labor ratio

    # a. solve for K and L
    initial_guess = np.array([KL_mid])

    if do_print: 
        print(f'starting at KL={KL_mid:.4f}')

    res = optimize.root(obj_ss, initial_guess, args=(model,))
    if do_print: 
        print('')
        print(res)
        print('')
    
    # b. final evaluations
    obj_ss(res.x,model)
    
    # c. show
    if do_print:
        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.K = :6.3f}')
        print(f'{ss.B = :6.3f}')
        print(f'{ss.A_hh = :6.3f}')
        print(f'{ss.L = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.G = :6.3f}')
        print(f'{ss.LG = :6.3f}')
        print(f'{ss.LY = :6.3f}')
        print(f'{ss.G/ss.Y = :6.3f}')
        print(f'{ss.tau = :6.3f}')
        print(f'{ss.chi = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_L = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')
        # print(f"Discounted utility (ss.du) = {np.sum(ss.du,axis=1).sum()}")