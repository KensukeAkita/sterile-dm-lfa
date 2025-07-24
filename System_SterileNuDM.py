import numpy as np
from numba import jit
import scipy as sp
from Constants import *
from Tolerances import *
from Setup_Bins import *
from Thermodynamics.DOF import grho_below120MeV, dgrhodT_below120MeV, grho_above120MeV, dgrhodT_above120MeV
from Asy_Redistribution import eqs_redist
from Thermodynamics.Thermo_quantities import Thermo_quantities

def System_SterileNuDM(T,sys_values,Gamma_a_Table,Gamma_a_anti_Table,s_ini,ms,Uas,Le_ini,Lmu_ini,Ltau_ini,flavor):

    La = sys_values[2*n] #(n-n_bar)/s_SM mixing with sterile nu

    #Redistribution of asymmetries (chemical potentials)

    x0 = np.array([0,0,0,0,0])

    ans =  sp.optimize.root(eqs_redist, x0, args = (T,La,Le_ini,Lmu_ini,Ltau_ini,flavor), method='hybr',tol=tol_root)  

    zeta_nue = ans.x[0] #chemical potential normalized by temperature, zeta = mu/T
    zeta_numu = ans.x[1]
    zeta_nutau = ans.x[2]

    zeta_B = ans.x[3]

    zeta_Q = ans.x[4]

    if (flavor == 'e'):

        zeta_nua = zeta_nue

    elif (flavor == 'mu'):

        zeta_nua = zeta_numu

    else:

        zeta_nua = zeta_nutau

    rho, s, P, Delta_rho, rho_nua, rho_a, Del_nua, Del_a, Del_nu_total, Del_c_total, Del_Q_QCD \
          = Thermo_quantities(T,zeta_nue,zeta_numu,zeta_nutau,zeta_B,zeta_Q,flavor)


    #Computing differential Delta_rho

    h = 1e-5
    
    T2 = T + h
    
    ans2 =  sp.optimize.root(eqs_redist, x0, args = (T2,La,Le_ini,Lmu_ini,Ltau_ini,flavor), method='hybr',tol=tol_root)  

    zeta_nue2 = ans2.x[0]
    zeta_numu2 = ans2.x[1]
    zeta_nutau2 = ans2.x[2]

    zeta_B2 = ans2.x[3]

    zeta_Q2 = ans2.x[4]

    rho2, s2, P2, Delta_rho2, rho_nua2, rho_a2, Del_nua2, Del_a2, Del_nu_total2, Del_c_total2, Del_Q_QCD2 \
        = Thermo_quantities(T2,zeta_nue2,zeta_numu2,zeta_nutau2,zeta_B2,zeta_Q2,flavor)

    dDelta_rhodT = (Delta_rho2 - Delta_rho)/h + 4/T*Delta_rho

    #Kinetic equations for sterile nu production

    kinetic_eqs = Kinetic_eq(T,sys_values,Gamma_a_Table,Gamma_a_anti_Table,zeta_nua,s_ini,rho,s,P,dDelta_rhodT,rho_nua, rho_a, Del_nua, Del_a, Del_nu_total, Del_c_total, Del_Q_QCD,ms,Uas)


    return kinetic_eqs




@jit(nopython=True,nogil=True,fastmath=True) 
def Kinetic_eq(T,sys_values,Gamma_a_Table,Gamma_a_anti_Table,zeta_nua,s_ini,rho,s,P,dDelta_rhodT,rho_nua, rho_a, Del_nua, Del_a, Del_nu_total, Del_c_total, Del_Q_QCD,ms,Uas):


    #Distribution for sterile neutrinos

    f_nus = np.zeros(n) 
    f_nus_anti = np.zeros(n)
 
    f_nus  = sys_values[0:n]        
    f_nus_anti   = sys_values[n:2*n]


    ###################################################
    #Kinetic equations for sterile neutrinos 
    ###################################################
    
    df_nusdt = np.zeros(n)

    df_nus_antidt = np.zeros(n)
    
    Integrand_drhosdt = 0

    Integrand_dLadx = 0

    rho_nus = 0



    TIdx = np.searchsorted(T_Table, T)
    
    for ni in range(n): #bin for neutrino spectrum 

        y_scale = y_sterile[ni] #y_scale = (s_ini/s)^(1/3)p/T
        dy_scale = dy_sterile[ni]
        p = (s/s_ini)**(1/3)*y_scale*T #physical momentum

        y = p/T #y=p/T

        #Matter potential

        Va = (2)**(1/2)*GF*(Del_nu_total + Del_nua + (-1/2 + 2*sW**2)*Del_c_total + Del_a + (1-2*sW**2)*Del_Q_QCD)*T**3 \
              - 8*(2)**(1/2)*GF*p/3*(rho_nua/mZ**2 + rho_a/mW**2)*T**4
        

        Va_anti = (2)**(1/2)*GF*(Del_nu_total + Del_nua + (-1/2 + 2*sW**2)*Del_c_total + Del_a + (1-2*sW**2)*Del_Q_QCD)*T**3 \
              + 8*(2)**(1/2)*GF*p/3*(rho_nua/mZ**2 + rho_a/mW**2)*T**4


        #Active neutrino opacity with interpolation 


        yIdx_tmp = (y-y_min)/(y_max-y_min)*(ni_Table-1)
        yIdx = int(yIdx_tmp) + 1
        
        Gamma_a_Slope_y2 = (Gamma_a_Table[yIdx][TIdx] - Gamma_a_Table[yIdx-1][TIdx])/(y_Table[yIdx] - y_Table[yIdx-1])
        Gamma_a_interpy2 = Gamma_a_Slope_y2*(y-y_Table[yIdx-1]) + Gamma_a_Table[yIdx-1][TIdx]
        
        Gamma_a_Slope_y1 = (Gamma_a_Table[yIdx][TIdx-1] - Gamma_a_Table[yIdx-1][TIdx-1])/(y_Table[yIdx] - y_Table[yIdx-1])
        Gamma_a_interpy1 = Gamma_a_Slope_y1*(y-y_Table[yIdx-1]) + Gamma_a_Table[yIdx-1][TIdx-1]
        
        Gamma_a_Slope_T = (Gamma_a_interpy2 - Gamma_a_interpy1)/(T_Table[TIdx] - T_Table[TIdx-1])
        Gamma_a = Gamma_a_Slope_T*(T-T_Table[TIdx -1]) + Gamma_a_interpy1
        
        Gamma_a_anti_Slope_y2 = (Gamma_a_anti_Table[yIdx][TIdx] - Gamma_a_anti_Table[yIdx-1][TIdx])/(y_Table[yIdx] - y_Table[yIdx-1])
        Gamma_a_anti_interpy2 = Gamma_a_anti_Slope_y2*(y-y_Table[yIdx-1]) + Gamma_a_Table[yIdx-1][TIdx]
        
        Gamma_a_anti_Slope_y1 = (Gamma_a_Table[yIdx][TIdx-1] - Gamma_a_Table[yIdx-1][TIdx-1])/(y_Table[yIdx] - y_Table[yIdx-1])
        Gamma_a_anti_interpy1 = Gamma_a_anti_Slope_y1*(y-y_Table[yIdx-1]) + Gamma_a_anti_Table[yIdx-1][TIdx-1]
        
        Gamma_a_anti_Slope_T = (Gamma_a_anti_interpy2 - Gamma_a_anti_interpy1)/(T_Table[TIdx] - T_Table[TIdx-1])
        Gamma_a_anti = Gamma_a_anti_Slope_T*(T-T_Table[TIdx -1]) + Gamma_a_anti_interpy1 

        
        #Oscillation Probability in matter
    
        Pas = 1/2*(ms**2/(2*p)*Uas)**2/((Va - ms**2/(2*p)*(1-Uas**2)**(1/2))**2 + (Gamma_a/2)**2)
        Pas_anti = 1/2*(ms**2/(2*p)*Uas)**2/((Va_anti + ms**2/(2*p)*(1-Uas**2)**(1/2))**2 + (Gamma_a_anti/2)**2)


        #Kinetic equations for sterile neutrinos

        df_nusdt[ni] = Gamma_a/2*Pas*(1/(np.exp(y - zeta_nua) + 1) - f_nus[ni])

        df_nus_antidt[ni] = Gamma_a_anti/2*Pas_anti*(1/(np.exp(y + zeta_nua) + 1) - f_nus_anti[ni])


        #Computing some integration quantities needed below
        
        Integrand_drhosdt += dy_scale*y_scale**3*(df_nusdt[ni] + df_nus_antidt[ni])

        Integrand_dLadx += -dy_scale*y_scale**2*(df_nusdt[ni] - df_nus_antidt[ni])

        rho_nus += dy_scale*y_scale**3*(f_nus[ni] + f_nus_anti[ni])


    rho_nus *=  1/(2*np.pi**2)*(s/s_ini)**(4/3)


    ###################################################
    #Lepton asymmetry evolution
    ###################################################

    dL_nuadt = np.zeros(1)

    dL_nuadt[0] =1/(2*np.pi**2*s_ini)*Integrand_dLadx #Evolution for entropy-scaled asymmetry 



    ###################################################
    #Photon temperature evolution (Continuity equation)
    ###################################################

    #DOF for energy density with zero chemical potential

    if (T<120):

        grho = grho_below120MeV(T)        
        dgrhodT = dgrhodT_below120MeV(T)

    else:

        grho = grho_above120MeV(T)
        dgrhodT = dgrhodT_above120MeV(T)

    #Hubble parameter

    rho_tot = (rho + rho_nus)*T**4 #Total energy density

    Hubble = 1/mpl*((8*np.pi)/3*rho_tot)**(1/2)

    dtdT = -(np.pi**2/30*(dgrhodT + 4*grho/T) + dDelta_rhodT)/(3*Hubble*(rho + P) + 1/(2*np.pi**2)*(s/s_ini)**(4/3)*Integrand_drhosdt)


    ###################################################
    #Converting d/dt-> d/dT
    ###################################################

    dL_nuadt[0] *= dtdT

    df_nusdt *= dtdT

    df_nus_antidt *= dtdT

    
    kinetic_eqs = np.concatenate((df_nusdt,df_nus_antidt,dL_nuadt))

    return kinetic_eqs
