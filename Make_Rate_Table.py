import numpy as np
import scipy as sp
from Constants import *
from QCD_GlobalParameters import * 
from Setup_Bins import *
from Tolerances import *
from Thermodynamics.DOF import *
from Interaction_rate.Interaction_rate_lepton import Interaction_rate_lepton
from Interaction_rate.Interaction_rate_quark import Interaction_rate_quark
from Interaction_rate.Interaction_rate_meson import Interaction_rate_meson
from Asy_Redistribution import eqs_redist

def Make_Table(La,Le_ini,Lmu_ini,Ltau_ini,flavor):
    
    #Array between TQCD and Tcut
        
    
    Gamma_a_intermedi = np.zeros((ni_Table,nT_intermedi))
    Gamma_a_anti_intermedi = np.zeros((ni_Table,nT_intermedi))
    
    #Array below TQCD
    
    Gamma_a_low = np.zeros((ni_Table,nT_low))
    Gamma_a_anti_low = np.zeros((ni_Table,nT_low))
    
    #Array above Tcut
    
    Gamma_a_high = np.zeros((ni_Table,nT_high))
    Gamma_a_anti_high = np.zeros((ni_Table,nT_high))
    
    #Total array
    
    Gamma_a_Table = np.zeros((ni_Table,nT_Table))
    Gamma_a_anti_Table = np.zeros((ni_Table,nT_Table))

    
    #Temperatures for interpolation (Artificial)
    
    
    T_interp  = np.zeros(4)
    
    T_interp[0] = LambdaQCD - 1
    T_interp[1] = LambdaQCD
    T_interp[2] = Tcut
    T_interp[3] = Tcut + 1 
    
    for ni in range(ni_Table):
        
        y = y_Table[ni]
        
        #Creating the interpolated reaction rate between TQCD and Tcut 
        
        Gamma_a_interp = np.zeros(4)
        Gamma_a_anti_interp = np.zeros(4)
        
        for nT in range(4):
            
            T = T_interp[nT]
            
            #Redistribution of asymmetries (chemical potentials)

            x0 = np.array([0,0,0,0,0])

            ans =  sp.optimize.root(eqs_redist, x0, args = (T,La,Le_ini,Lmu_ini,Ltau_ini,flavor), method='hybr',tol=tol_root)  
            
            zeta_nue = ans.x[0] #chemical potential normalized by temperature, zeta = mu/T
            zeta_numu = ans.x[1]
            zeta_nutau = ans.x[2]

            zeta_e = ans.x[0] - ans.x[4]
            zeta_mu = ans.x[1] - ans.x[4]
            zeta_tau = ans.x[2] - ans.x[4]
        
            zeta_B = ans.x[3]
            zeta_Q = ans.x[4]

            zeta_U = 1/3*zeta_B + 2/3*zeta_Q
            zeta_D = 1/3*zeta_B - 1/3*zeta_Q
            
            if (flavor == 'e'):

                zeta_nua = zeta_nue
                zeta_a = zeta_e

            elif (flavor == 'mu'):

                zeta_nua = zeta_numu
                zeta_a = zeta_mu


            else:

                zeta_nua = zeta_nutau
                zeta_a = zeta_tau
            
            Gamma_lepton, Gamma_anti_lepton = Interaction_rate_lepton(T,y,zeta_nue,zeta_numu,zeta_nutau,zeta_e,zeta_mu,zeta_tau,flavor) 
            Gamma_quark, Gamma_anti_quark  = Interaction_rate_quark(T,y,zeta_nua,zeta_a,zeta_U,zeta_D,flavor)
            Gamma_meson, Gamma_anti_meson  =  Interaction_rate_meson(T,y,zeta_nua,zeta_a,zeta_Q,zeta_U,zeta_D,flavor)
            
            Gamma_a_interp[nT] = (Gamma_lepton + Gamma_quark + Gamma_meson)
            Gamma_a_anti_interp[nT] = (Gamma_anti_lepton + Gamma_anti_quark + Gamma_anti_meson)
            
        f_Gamma_a = sp.interpolate.interp1d(T_interp,Gamma_a_interp,kind='cubic')
        f_Gamma_a_anti = sp.interpolate.interp1d(T_interp,Gamma_a_anti_interp,kind='cubic')
        
        for nT in range(nT_intermedi):
        
            T =T_intermedi[nT]
    
            Gamma_a_intermedi[ni,nT] = f_Gamma_a(T)
            Gamma_a_anti_intermedi[ni,nT] = f_Gamma_a_anti(T)
            
        #Creating the reaction rate below TQCD
        
        for nT in range(nT_low):
            
            T = T_low[nT]
            
            #Redistribution of asymmetries
            
            x0 = np.array([0,0,0,0,0])

            ans =  sp.optimize.root(eqs_redist, x0, args = (T,La,Le_ini,Lmu_ini,Ltau_ini,flavor), method='hybr',tol=tol_root)  
            
            zeta_nue = ans.x[0]
            zeta_numu = ans.x[1]
            zeta_nutau = ans.x[2]

            zeta_e = ans.x[0] - ans.x[4]
            zeta_mu = ans.x[1] - ans.x[4]
            zeta_tau = ans.x[2] - ans.x[4]
        
            zeta_B = ans.x[3]
            zeta_Q = ans.x[4]

            zeta_U = 1/3*zeta_B + 2/3*zeta_Q
            zeta_D = 1/3*zeta_B - 1/3*zeta_Q
            
            if (flavor == 'e'):

                zeta_nua = zeta_nue
                zeta_a = zeta_e

            elif (flavor == 'mu'):

                zeta_nua = zeta_numu
                zeta_a = zeta_mu


            else:

                zeta_nua = zeta_nutau
                zeta_a = zeta_tau
            
            Gamma_lepton, Gamma_anti_lepton = Interaction_rate_lepton(T,y,zeta_nue,zeta_numu,zeta_nutau,zeta_e,zeta_mu,zeta_tau,flavor) 
            Gamma_quark, Gamma_anti_quark  = Interaction_rate_quark(T,y,zeta_nua,zeta_a,zeta_U,zeta_D,flavor)
            Gamma_meson, Gamma_anti_meson  =  Interaction_rate_meson(T,y,zeta_nua,zeta_a,zeta_Q,zeta_U,zeta_D,flavor)
            
            Gamma_a_low[ni,nT] = (Gamma_lepton + Gamma_quark + Gamma_meson)
            Gamma_a_anti_low[ni,nT] = (Gamma_anti_lepton + Gamma_anti_quark + Gamma_anti_meson)
            
        #Creating the reaction rate above Tcut
        
        for nT in range(nT_high):
            
            T = T_high[nT]
            
            #Redistribution of asymmetries
            
            x0 = np.array([0,0,0,0,0])

            ans =  sp.optimize.root(eqs_redist, x0, args = (T,La,Le_ini,Lmu_ini,Ltau_ini,flavor), method='hybr',tol=tol_root)  
            
            zeta_nue = ans.x[0]
            zeta_numu = ans.x[1]
            zeta_nutau = ans.x[2]

            zeta_e = ans.x[0] - ans.x[4]
            zeta_mu = ans.x[1] - ans.x[4]
            zeta_tau = ans.x[2] - ans.x[4]
        
            zeta_B = ans.x[3]
            zeta_Q = ans.x[4]

            zeta_U = 1/3*zeta_B + 2/3*zeta_Q
            zeta_D = 1/3*zeta_B - 1/3*zeta_Q
            
            if (flavor == 'e'):

                zeta_nua = zeta_nue
                zeta_a = zeta_e

            elif (flavor == 'mu'):

                zeta_nua = zeta_numu
                zeta_a = zeta_mu


            else:

                zeta_nua = zeta_nutau
                zeta_a = zeta_tau
            
            Gamma_lepton, Gamma_anti_lepton = Interaction_rate_lepton(T,y,zeta_nue,zeta_numu,zeta_nutau,zeta_e,zeta_mu,zeta_tau,flavor) 
            Gamma_quark, Gamma_anti_quark  = Interaction_rate_quark(T,y,zeta_nua,zeta_a,zeta_U,zeta_D,flavor)
            Gamma_meson, Gamma_anti_meson  =  Interaction_rate_meson(T,y,zeta_nua,zeta_a,zeta_Q,zeta_U,zeta_D,flavor)
            
            Gamma_a_high[ni,nT] = (Gamma_lepton + Gamma_quark + Gamma_meson)
            Gamma_a_anti_high[ni,nT] = (Gamma_anti_lepton + Gamma_anti_quark + Gamma_anti_meson)
            
        #Computing total reaction rate
        
        for nT in range(nT_Table):
            
            if (nT < nT_low):
                
                Gamma_a_Table[ni,nT] =  Gamma_a_low[ni,nT]
                Gamma_a_anti_Table[ni,nT] =  Gamma_a_anti_low[ni,nT]
                
            elif (nT >= nT_low) and (nT < nT_low + nT_intermedi):
                
                Gamma_a_Table[ni,nT] =  Gamma_a_intermedi[ni,nT-nT_low]
                Gamma_a_anti_Table[ni,nT] =  Gamma_a_anti_intermedi[ni,nT-nT_low]
                
            else:
                
                Gamma_a_Table[ni,nT] =  Gamma_a_high[ni,nT-nT_low-nT_intermedi]
                Gamma_a_anti_Table[ni,nT] =  Gamma_a_anti_high[ni,nT-nT_low-nT_intermedi]
        

    return Gamma_a_Table, Gamma_a_anti_Table


