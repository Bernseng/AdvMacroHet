import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,l):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in nb.prange(par.Nfix):

        chi = par.chi[i_fix % 2] / par.eta_grid[i_fix]
        phi = par.phi[i_fix % 2] / par.eta_grid[i_fix]

        # a. solve step
        for i_z in nb.prange(par.Nz):
        
            ## i. labor supply
            # Adjusted labor supply according to ability and labor type

            # l[i_fix,i_z,:] = par.eta_grid[i_fix] * par.z_grid[i_z]
            # if par.eta_grid[i_fix] == 0:  # low ability
            #     l[i_fix,i_z,:] = par.z_grid[i_z] * (2/3 * phi_0 + 1/3 * phi_1)
            # else:  # high ability
            #     l[i_fix,i_z,:] = par.z_grid[i_z] * (1/3 * phi_0 + 2/3 * phi_1)
 

            l[i_fix,i_z,:] = par.z_grid[i_z] * par.eta_grid[i_fix] * chi * phi
            
            # l[i_fix,i_z,:] = par.z_grid[i_z]*par.eta_grid[i_fix]
            # if par.eta_grid[i_fix] == 0:
            # ## ii. cash-on-hand
            #     m = (1+r-par.delta)*par.a_grid + w*l[i_fix,i_z,:]*phi_0
            # else:
            #     m = (1+r-par.delta)*par.a_grid + w*l[i_fix,i_z,:]*phi_1

            m = (1+r-par.delta)*par.a_grid + w*l[i_fix,i_z,:]


            # iii. EGM
            c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid # current consumption + end-of-period assets
            
            # iv. interpolation to fixed grid
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        v_a = (1+r-par.delta)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a