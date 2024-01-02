import numpy as np
import numba as nb

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # not used today: .sim and .path
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w0','w1','phi0','phi1'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c','l0','l1','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['Gamma','phi0','phi1'] # exogenous shocks
        self.unknowns = ['K','L0','L1'] # endogenous unknowns
        self.targets = ['clearing_A','clearing_L0','clearing_L1'] # targets = 0
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'hh', # household block
            'blocks.market_clearing']
        
        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 6 # number of fixed discrete states
        par.Nz = 7 # number of stochastic discrete states (here productivity)
        par.phi0_ss = 1.0 # steady state productivity of labor type 0
        par.phi1_ss = 2.0 # steady state productivity of labor type 1

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta_mean = 0.975 # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.010 # discount factor, width, range is [mean-width,mean+width]

        # b. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # c. production and investment
        par.alpha = 0.36 # cobb-douglas
        par.delta = 0.10 # depreciation rate
        par.Gamma_ss = 1.0 # direct approach: technology level in steady state
        par.epsilon = 1.0
        par.nu = 0.5

        # d. shocks
        par.jump_phi1 = 0.10 # initial jump
        par.rho_phi1 = 0.90 # AR(1) coefficient
        par.std_phi1 = 0.01 # std. of innovation

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.01
        par.w_ss_target = 1.0

        # h. misc.
        par.T = 500 # length of transition path
        par.simT = 2_000 # length of simulation 

        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system

        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_hh = False # call solve_hh_backwards in Python-model
        par.py_block = True # call blocks in Python-model
        par.full_z_trans = False # let z_trans vary over endogenous states
        par.warnings = True # print warnings if nans are encountered
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids   
        par.Nbeta = 3
        par.beta_grid = np.zeros(par.Nfix)
        par.eta0_grid = np.zeros(par.Nfix)
        par.eta1_grid = np.zeros(par.Nfix)

        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    def v_ss(self):
        """ social welfare in transition path """

        par = self.par
        path = self.path
        for i_fix in nb.prange(par.Nfix):
            v = np.sum([par.beta_grid[i_fix]**t*np.sum(path.u[t,i_fix]*path.D[t,i_fix]/np.sum(path.D[t,i_fix])) for t in range(par.T)])
        return v