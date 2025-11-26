import numpy as np
import cvxpy as cvx
import control as ctrl
import scipy.linalg as linalg
from fct.build_interconnection import build_appended_system
from fct.build_interconnection import combine_filters_different_nonlinearities


def compute_rho_for_acc_admm(m, L, n_ZF, algo, v1=None, v2=None, rho_max=1.3, eps=1e-6, alpha=1):
    """
    Computes the worst-case convergence rate (ρ_ADMM) for the ADMM sector method.

    Parameters:
        m (float): Strong convexity parameter.
        L (float): Smoothness parameter.
        rho_max (float): Initial maximum value of ρ.
        eps (float): Small positive tolerance.
        alpha (float): ADMM relaxation parameter.

    Returns:
        float: Worst-case convergence rate (ρ_ADMM).
    """
    rho_min = 0
    rho_tol = 1e-3
    kappa = L / m
    lower_sector = m
    upper_sector = L


    if algo=='ADMM':
        v1 = 1 
        v2 = 0
        alpha = 1
    elif algo=='OR-ADMM':
        v1 = 1 
        v2 = 0

    elif algo=='A-ADMM (NM)':
        v1 = 1/L 
        v2 = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1) 
        alpha = 1

    
    elif algo=='A-ADMM (TM)':
        gamma = 1-(1/np.sqrt(kappa))
        v1 = (1+gamma)/L 
        v2 = 1*(gamma**2)/(2-gamma)
        alpha = 1

    elif algo=='A-ADMM (TM, λ-damped)':
        gamma = 1-(1/np.sqrt(kappa))
        v1 = (1+gamma)/L 
        v2 = 1*(gamma**2)/(2-gamma)
        damp = 0.1
        alpha = 1

    elif algo == 'A-ADMM (GS)':
        if v1 is None or v2 is None:
            raise ValueError("v1 and v2 must be provided for 'A-ADMM (GS)'.")
        alpha = 1

    elif algo == 'OR-A-ADMM (GS)':
        if v1 is None or v2 is None:
            raise ValueError("v1 and v2 must be provided for 'OR-A-ADMM (GS)'.")
        
    else:
        raise ValueError(f"Wrong algorithm has been passed: '{algo}'")


    # Define the system (Plant)
    while (rho_max - rho_min > rho_tol):

        rho = (rho_min + rho_max) / 2


        if algo=='A-ADMM (TM, λ-damped)':
            AA = np.array([[0, 0, 1, 0], 
                           [0, 0, 0, 1], 
                           [0, 0, damp, 0],
                           [-v2*(alpha-1), -v2, (alpha-1)*(1+v2), 1+v2]])
            BB = np.array([[0, 0],
                           [0, 0],
                           [0, -v1*damp],
                           [alpha*v1, -v1]])
            
        else:
            AA = np.array([[0, 0, 1, 0], 
                           [0, 0, 0, 1], 
                           [0, 0, 0, 0],
                           [-v2*(alpha-1), -v2, (alpha-1)*(1+v2), 1+v2]])
            BB = np.array([[0, 0],
                           [0, 0],
                           [0, -v1],
                           [alpha*v1, -v1]])
            
        CC = np.array([[v2,             v2,          -(1+v2), -(1+v2)],
                       [-v2*(alpha-1), -v2, (alpha-1)*(1+v2),    1+v2]])
        DD = np.array([[-v1,        0], 
                       [alpha*v1, -v1]])
        
        sys = ctrl.ss(AA, BB, CC, DD, dt=1, name='G')
        
        ### ACAUSAL ZAMES-FALB IQC ###  
        # Filter is implemented as \Psi(z) in Van Scoy et al (2022) - Absolute Stability via Interpolation and Lifting (above eq. (8))
        # Filter is implemented symmetrically (= causal degree is equal to acausal degree)
        #
        # Psi(z) = [1        0]
        #          [z^-1     0]
        #          [...    ...]
        #          [z^-n_ZF  0]
        #          [0        1]
        #          [0     z^-1]
        #          [...    ...]
        #          [0  z^-n_ZF]
        #          
        # For [m,L] sector, we take Psi(z)*S_mL

        A_psi = np.diag(np.ones(n_ZF-1), k=-1)
        A_psi = linalg.block_diag(A_psi, A_psi)
        B_psi_v, B_psi_w = np.zeros((n_ZF, 2)), np.zeros((n_ZF, 2))
        B_psi_v[0,0] = 1
        B_psi_w[0,1] = 1
        B_psi = np.vstack([B_psi_v, B_psi_w])
        C_psi = np.vstack([np.zeros((1, n_ZF)),
                           np.eye(n_ZF)])
        C_psi = linalg.block_diag(C_psi, C_psi)
        D_psi_v, D_psi_w = np.zeros((n_ZF+1, 2)), np.zeros((n_ZF+1, 2))
        D_psi_v[0,0] = 1
        D_psi_w[0,1] = 1
        D_psi = np.vstack([D_psi_v, D_psi_w])

        Sml = np.asarray([[upper_sector, -1],
                          [-lower_sector, 1]])

        Psi_zInf = ctrl.ss(A_psi, B_psi, C_psi, D_psi, dt=1, name='Psi_zInf')
        Psi_mL   = ctrl.ss(A_psi, B_psi@Sml, C_psi, D_psi@Sml, dt=1, name='Psi_mL')

        unique_filter = combine_filters_different_nonlinearities(2, Psi_mL, Psi_zInf)
        sys_ap = build_appended_system(sys,unique_filter)
        A1, B1, C1, D1 = ctrl.ssdata(sys_ap)

        nx = A1.shape[0]
        nu = B1.shape[1]

        # Define variables and LMI constraints
        P = cvx.Variable((nx, nx), symmetric=True)
        pi_mL_0  = cvx.Variable((1,1))
        pi_mL_c  = cvx.Variable((n_ZF,1))
        pi_mL_ac = cvx.Variable((n_ZF,1))
        pi_zInf_0  = cvx.Variable((1,1))
        pi_zInf_c  = cvx.Variable((n_ZF,1)) 
        pi_zInf_ac = cvx.Variable((n_ZF,1))

        N_mL = cvx.bmat([[pi_mL_0,                pi_mL_c.T],
                         [pi_mL_ac,  np.zeros((n_ZF, n_ZF))]])
        N_zInf = cvx.bmat([[pi_zInf_0,              pi_zInf_c.T],
                           [pi_zInf_ac,  np.zeros((n_ZF, n_ZF))]])
        
        Multiplier_mL = cvx.bmat([[np.zeros((n_ZF+1,n_ZF+1)),                     N_mL.T],
                                  [N_mL,                       np.zeros((n_ZF+1,n_ZF+1))]])

        Multiplier_zInf = cvx.bmat([[np.zeros((n_ZF+1,n_ZF+1)),                   N_zInf.T],
                                    [N_zInf,                     np.zeros((n_ZF+1,n_ZF+1))]])

        Multiplier = cvx.bmat([[Multiplier_mL,                     np.zeros((2*(n_ZF+1),2*(n_ZF+1)))],
                               [np.zeros((2*(n_ZF+1),2*(n_ZF+1))), Multiplier_zInf]])
        
        LMI1 = cvx.bmat([[A1, B1]]).T @ P @ cvx.bmat([[A1, B1]])
        LMI2 = cvx.bmat([[P, np.zeros((nx, nu))], [np.zeros((nu, nx)), np.zeros((nu, nu))]])
        LMI3 = cvx.bmat([[C1, D1]]).T @ Multiplier @ cvx.bmat([[C1, D1]])

        LMI = LMI1 - rho**2 * LMI2 + LMI3

        constraints_LMI = [P >> eps * np.eye(nx), LMI << 0]

        # Condition of the multipliers pi are chosen as in Scherer (2023) - Robust Exponential Stability and Inviariance Guarantees with General Dynamic OZF Multipliers (eq. (26))
        vec_rho = (rho**2) ** np.arange(1, n_ZF+1)
        vec_rho_inv = (rho**-2) ** np.arange(1, n_ZF+1)
        constraints_ZF1 = [pi_mL_0 >= 0, 
                           pi_mL_c <= 0, 
                           pi_mL_ac <= 0,
                           pi_mL_0 + vec_rho @ pi_mL_c + vec_rho_inv @ pi_mL_ac >= 0,
                           pi_mL_0 + vec_rho_inv @ pi_mL_c + vec_rho @ pi_mL_ac >= 0]
        constraints_ZF2 = [pi_zInf_0 >= 0, 
                           pi_zInf_c <= 0,
                           pi_zInf_ac <= 0,
                           pi_zInf_0 + vec_rho @ pi_zInf_c + vec_rho_inv @ pi_zInf_ac >= 0,
                           pi_zInf_0 + vec_rho_inv @ pi_zInf_c + vec_rho @ pi_zInf_ac >= 0]

        constraints = constraints_LMI + constraints_ZF1 + constraints_ZF2

        problem = cvx.Problem(cvx.Minimize(0), constraints=constraints)
        try:
            problem.solve(solver=cvx.MOSEK)
        except cvx.SolverError:
            pass

        # Update interval based on feasibility
        if problem.status == cvx.OPTIMAL:
            rho_max = rho
        else:
            rho_min = rho

    return rho_max
