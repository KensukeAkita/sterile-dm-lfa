"""
This file implements the neutrino interaction rates with mesons for a fixed momentum, i.
"""

import numpy as np
from numba import jit
from Constants import *
from Setup_Grids import *

#In this version we include both fusion and scattering processes with heavy mesons

@jit(nopython=True,nogil=True,fastmath=True)
def Interaction_rate_meson(T,i,zeta_nua,zeta_a,zeta_Q,zeta_U,zeta_D,flavor):

    Coll = 0
    Coll_bar = 0

    xe = me/T
    xmu = mmu/T
    xtau = mtau/T

    xpi0 = mpi0/T
    xpic = mpic/T
    xK0 = mK0/T
    xKc = mKc/T
    xeta = meta/T
    xrho0 = mrho0/T
    xrhoc = mrhoc/T
    xKc892 = mKc892/T
    xomega = momega/T

    zeta_pic = -zeta_Q #negatively charged pion #chemical potential normalized by temperature, zeta = mu/T
    zeta_mesonc = -zeta_Q #negatively charged meson
    zeta_meson0 = 0

    Vmatrix = [[Vud,Vus],[Vcd,Vcs]]

    if (flavor == 'e'): #0: sterile nu mixing with nue, 1: sterile nu mixing with numu, 2:sterile nu mixing with nutau

        xa = xe

        Gamma_pi = 3.110*10**(-18) #MeV
        Gamma_K  = 8.41*10**(-19) #MeV
        Gamma_rho0 = 9.78*10**(-12)
        Gamma_rhoc = 7.00*10**(-11)
        Gamma_K892 = 5.45*10**(-12) 
        Gamma_omega = 7*10**(-13)

        

    elif (flavor == 'mu'):

        xa = xmu

        Gamma_pi = 2.528*10**(-14)  #MeV
        Gamma_K  = 3.38*10**(-14) #MeV
        Gamma_rho0 = 9.78*10**(-12)
        Gamma_rhoc = 6.80*10**(-11)
        Gamma_K892 = 5.33*10**(-12) 
        Gamma_omega = 7*10**(-13)


    else:

        xa = xtau

        Gamma_rho0 = 9.78*10**(-12)
        Gamma_omega = 7*10**(-13)




    #Collision terms #The current version only include pions.    
    for nk in range(n1): #bin for an integration in the collision term

        k = y_min1 + dy1*nk
        
        for nj in range(n1): #bin for an integration in the collision term
            
            weight_simps=coe_simps[nk]*coe_simps[nj]
            
            j = y_min1 + dy1*nj 

            #meson contribution above T<Lamda_QCD
            if (T<Tcut-10):


                #s-channel (nua +a^+ <->)

                xj = xa
                zeta_j = zeta_a
                Ei = i
                Ej = (j**2 + xa**2)**(1/2)

                for ny in range(n1):

                    y = -1 + 2/(n1-1)*(ny-1) #This is cos theta12.

                    s = (xj**2 + 2*Ei*Ej - 2*i*j*y)*T**2

                    if (s > 1e6): 

                        #4-point scattering between neutrinos and free quarks

                        #nua + a^+ <-> U + D_bar (s-channel)

                        for upquark in range(2):

                            if (upquark ==0):

                                xU = mu/T

                            else:

                                xU = mc/T

                            for downquark in range(2):

                                if (downquark ==0):

                                    xD = md/T

                                else:

                                    xD = ms/T

                                V = Vmatrix[upquark][downquark]

                                xk = xU
                                xl = xD

                                zeta_k = zeta_U
                                zeta_l = zeta_D

                                Ek = (k**2 + xk**2)**(1/2)
                                El = i + Ej - Ek

                                Q = xj**2 + xk**2 - xl**2

                                if (El > xl):

                                    f_j = 1/(np.exp(Ej - zeta_j)+1) 
                                    f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                                    f_k = 1/(np.exp(Ek - zeta_k)-1) 
                                    f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                                    f_l = 1/(np.exp(El - zeta_l)-1) 
                                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                                    #We will use the method in arXiv:astro-ph/9506015

                                    F = f_j_bar*(1-f_l_bar)*(1-f_k)
                                    F_bar = f_j*(1-f_l)*(1-f_k_bar)

                                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                                    kap = i**2 + k**2

                                    A = -4*k**2*(2*i*j + i**2 + j**2)
                                    B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                                    C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                                    if (B**2-4*A*C>0):

                                        Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                        Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                        if (Bet<1) and (Bet>-1):

                                            if (Alp<1) and (Alp>-1):

                                                #p1*p2

                                                pij = Ei*Ej - i*j*y

                                                #(p2*p3)(p1*p4)

                                                p_jk_il_x2 = i**2*k**2

                                                p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                                p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                                fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                                fac2 = 384*V**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                                                Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*p_jk_il_x2 \
                                                        + 1/2*(Alp + Bet)*p_jk_il_x \
                                                        + p_jk_il_0)
                                        

                                                Coll += Coll_tmp*F

                                                Coll_bar += Coll_tmp*F_bar


                    else: 

                        #nua + a^+ <-> pi^+ + pi^0 (s-channel)

                        xk = xpic
                        xl = xpi0

                        zeta_k = zeta_pic
                        zeta_l = 0

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)+1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            #We will use the method in arXiv:astro-ph/9506015

                            F = f_j_bar*(1+f_l_bar)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 8*Vud**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar

                        #nua + a^+ <-> K^+ + K^0_bar (s-channel)

                        xk = xKc
                        xl = xK0

                        zeta_k = zeta_pic
                        zeta_l = 0

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)+1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            #We will use the method in arXiv:astro-ph/9506015

                            F = f_j_bar*(1+f_l_bar)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 8*Vud**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar

                        #nua + a^+ <-> pi^+ + K^0 (s-channel)

                        xk = xpic
                        xl = xK0

                        zeta_k = zeta_pic
                        zeta_l = 0

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)+1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            #We will use the method in arXiv:astro-ph/9506015

                            F = f_j_bar*(1+f_l_bar)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar

                        #nua + a^+ <-> K^+ + pi^0 (s-channel)

                        xk = xKc
                        xl = xpi0

                        zeta_k = zeta_pic
                        zeta_l = 0

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)+1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            #We will use the method in arXiv:astro-ph/9506015

                            F = f_j_bar*(1+f_l_bar)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar

                        #nua + a^+ <-> K^+ + eta (s-channel)

                        xk = xKc
                        xl = xeta

                        zeta_k = zeta_pic
                        zeta_l = 0

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)+1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            #We will use the method in arXiv:astro-ph/9506015

                            F = f_j_bar*(1+f_l_bar)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar



                #s-channel (nua + nua_bar <->)

                xj = 0
                zeta_j = zeta_nua
                Ei = i
                Ej = j

                for ny in range(n1):

                    y = -1 + 2/(n1-1)*(ny-1) #This is cos theta12.

                    s = (2*Ei*Ej - 2*i*j*y)*T**2

                    if (s > 1e6): 

                        #4-point scattering between neutrinos and free quarks

                        for upquark in range(2):

                            if (upquark ==0):

                                xU = mu/T

                            else:

                                xU = mc/T

                            for downquark in range(2):

                                if (downquark ==0):

                                    xD = md/T

                                else:

                                    xD = ms/T

                                
                                #nua + nua^+ <-> U + U_bar (s-channel)

                                xk = xU
                                xl = xU

                                zeta_k = zeta_U
                                zeta_l = zeta_U

                                Ek = (k**2 + xk**2)**(1/2)
                                El = i + Ej - Ek

                                Q = xj**2 + xk**2 - xl**2

                                if (El > xl):

                                    f_j = 1/(np.exp(Ej - zeta_j)+1) 
                                    f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                                    f_k = 1/(np.exp(Ek - zeta_k)-1) 
                                    f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                                    f_l = 1/(np.exp(El - zeta_l)-1) 
                                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                                    #We will use the method in arXiv:astro-ph/9506015

                                    F = f_j_bar*(1-f_l_bar)*(1-f_k)
                                    F_bar = f_j*(1-f_l)*(1-f_k_bar)

                                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                                    kap = i**2 + k**2

                                    A = -4*k**2*(2*i*j + i**2 + j**2)
                                    B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                                    C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                                    if (B**2-4*A*C>0):

                                        Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                        Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                        if (Bet<1) and (Bet>-1):

                                            if (Alp<1) and (Alp>-1):

                                                #p1*p2

                                                pij = Ei*Ej - i*j*y

                                                
                                                #(p2*p4)(p1*p3)

                                                p_jl_ik_x2 = i**2*k**2

                                                p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                                p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                                #(p2*p3)(p1*p4)

                                                p_jk_il_x2 = i**2*k**2

                                                p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                                p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                                #(p1*p2)*mu**2

                                                pij_0 = pij*xk**2

                                                fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                                fac2 = 32/3*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps*1/2 #1/2: avoiding double count in the loops

                                                Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(16*gR**2*p_jl_ik_x2 + (3-4*gR)**2*p_jk_il_x2) \
                                                        + 1/2*(Alp + Bet)*(16*gR**2*p_jl_ik_x + (3-4*gR)**2*p_jk_il_x) \
                                                        + 16*gR**2*p_jl_ik_0 + (3-4*gR)**2*p_jk_il_0 + 4*gR*(4*gR-3)*pij_0) 
                                        

                                                Coll += Coll_tmp*F

                                                Coll_bar += Coll_tmp*F_bar




                                #nua + nua^+ <-> D + D_bar (s-channel)

                                xk = xD
                                xl = xD

                                zeta_k = zeta_D
                                zeta_l = zeta_D

                                Ek = (k**2 + xk**2)**(1/2)
                                El = i + Ej - Ek

                                Q = xj**2 + xk**2 - xl**2

                                if (El > xl):

                                    f_j = 1/(np.exp(Ej - zeta_j)+1) 
                                    f_j_bar = 1/(np.exp(Ej + zeta_j)+1) 
                                    f_k = 1/(np.exp(Ek - zeta_k)-1) 
                                    f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                                    f_l = 1/(np.exp(El - zeta_l)-1) 
                                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                                    #We will use the method in arXiv:astro-ph/9506015

                                    F = f_j_bar*(1-f_l_bar)*(1-f_k)
                                    F_bar = f_j*(1-f_l)*(1-f_k_bar)

                                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                                    kap = i**2 + k**2

                                    A = -4*k**2*(2*i*j + i**2 + j**2)
                                    B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                                    C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                                    if (B**2-4*A*C>0):

                                        Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                        Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                        if (Bet<1) and (Bet>-1):

                                            if (Alp<1) and (Alp>-1):

                                                #p1*p2

                                                pij = Ei*Ej - i*j*y

                                                
                                                #(p2*p4)(p1*p3)

                                                p_jl_ik_x2 = i**2*k**2

                                                p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                                p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                                #(p2*p3)(p1*p4)

                                                p_jk_il_x2 = i**2*k**2

                                                p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                                p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                                #(p1*p2)*mu**2

                                                pij_0 = pij*xk**2

                                                fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                                fac2 = 32/3*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps*1/2 #1/2: avoiding double count in the loops

                                                Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(4*gR**2*p_jl_ik_x2 + (3-2*gR)**2*p_jk_il_x2) \
                                                        + 1/2*(Alp + Bet)*(4*gR**2*p_jl_ik_x + (3-2*gR)**2*p_jk_il_x) \
                                                        + 4*gR**2*p_jl_ik_0 + (3-2*gR)**2*p_jk_il_0 + 2*gR*(2*gR-3)*pij_0) 
                                        

                                                Coll += Coll_tmp*F

                                                Coll_bar += Coll_tmp*F_bar


                    else: 

                        #nua + nua_bar <-> pi^+ + pi^- (s-channel)

                        xk = xpic
                        xl = xpic

                        zeta_k = zeta_pic
                        zeta_l = zeta_pic

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)-1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            F = f_j_bar*(1+f_l)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l_bar)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 4*(1-2*sW)**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar

                        #nua + nua_bar <-> K^+ + K^- (s-channel)

                        xk = xKc
                        xl = xKc

                        zeta_k = zeta_pic
                        zeta_l = zeta_pic

                        Ek = (k**2 + xk**2)**(1/2)
                        El = i + Ej - Ek

                        Q = xj**2 + xk**2 - xl**2

                        if (El > xl):

                            f_j = 1/(np.exp(Ej - zeta_j)-1) 
                            f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                            f_k = 1/(np.exp(Ek - zeta_k)-1) 
                            f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                            f_l = 1/(np.exp(El - zeta_l)-1) 
                            f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                            F = f_j_bar*(1+f_l)*(1+f_k_bar)
                            F_bar = f_j*(1+f_l_bar)*(1+f_k)

                            gam = Ei*Ej - Ei*Ek - Ej*Ek
                            kap = i**2 + k**2

                            A = -4*k**2*(2*i*j + i**2 + j**2)
                            B = 4*(2*i*j**2*k + 2*i**2*j*k - k*(i+j)*(2*gam + Q))
                            C = 4*(-j**2*kap + i*j*(2*gam + Q) - gam**2 - gam*Q - Q**2 + j**2*k**2)

                            if (B**2-4*A*C>0):

                                Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                                Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                                if (Bet<1) and (Bet>-1):

                                    if (Alp<1) and (Alp>-1):

                                        #p1*p2

                                        pij = Ei*Ej - i*j*y

                                        #(p2*p3)(p1*p3)

                                        p_jk_ik_x2 = -i**2*k**2

                                        p_jk_ik_x = -i*k*(pij +Q/2)

                                        p_jk_ik_0 = Ei*Ek*(pij -Ei*Ek + Q/2)

                                        #(p2*p4)(p1*p3)

                                        p_jl_ik_x2 = i**2*k**2

                                        p_jl_ik_x = -i*k*(2*Ei*Ek + xj**2 -Q/2)

                                        p_jl_ik_0 = Ei*Ek*(Ei*Ek + xj**2 - Q/2)

                                        #(p2*p3)(p1*p4)

                                        p_jk_il_x2 = i**2*k**2

                                        p_jk_il_x = i*k*(2*pij - 2*Ei*Ek + Q/2)

                                        p_jk_il_0 = (pij - Ei*Ek + Q/2)*(pij - Ei*Ek)

                                        #(p2*p4)(p1*p4)

                                        p_jl_il_x2 = -i**2*k**2

                                        p_jl_il_x = -i*k*(pij -2*Ei*Ek - xj**2 + Q/2)

                                        p_jl_il_0 = (Ei*Ek + xj**2 - Q/2)*(pij - Ei*Ek)

                                        #(p1*p2)(p3*p4)

                                        p_ij_kl_0 = pij*(pij - xk**2 + Q/2)

                                        #(p1*p2)*(m3**2 + m4**2)

                                        pij_0 = pij*(xk**2 + xl**2)

                                        fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                        fac2 = 4*(1-2*sW)**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 


                                        Coll_tmp = fac2*fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_jk_ik_x2 - p_jl_ik_x2 - p_jk_il_x2 +p_jl_il_x2) \
                                                        + 1/2*(Alp + Bet)*(p_jk_ik_x - p_jl_ik_x - p_jk_il_x + p_jl_il_x) \
                                                        + (p_jk_ik_0 - p_jl_ik_0 - p_jk_il_0 + p_jl_il_0 + p_ij_kl_0 -1/2*pij_0))
                                        

                                        Coll += Coll_tmp*F

                                        Coll_bar += Coll_tmp*F_bar




                #nua + pi^+- <-> nua + pi^+- (t-channel)

                xj = xpic
                xk = 0
                xl = xpic

                zeta_j = zeta_pic
                zeta_k = zeta_nua
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)-1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F_pl = f_j_bar*(1+f_l_bar)*(1-f_k)
                    F_bar_pl = f_j*(1+f_l)*(1-f_k_bar)

                    F_mi = f_j*(1+f_l)*(1-f_k)
                    F_bar_mi = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 4*(1-2*sW)**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*(F_pl + F_mi)
                    Coll_bar += overall_fac*Coll_tmp*(F_bar_pl + F_bar_mi)

                #nua + K^+- <-> nua + K^+- (t-channel)

                xj = xKc
                xk = 0
                xl = xKc

                zeta_j = zeta_pic
                zeta_k = zeta_nua
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)-1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)-1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F_pl = f_j_bar*(1+f_l_bar)*(1-f_k)
                    F_bar_pl = f_j*(1+f_l)*(1-f_k_bar)

                    F_mi = f_j*(1+f_l)*(1-f_k)
                    F_bar_mi = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 4*(1-2*sW)**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*(F_pl + F_mi)
                    Coll_bar += overall_fac*Coll_tmp*(F_bar_pl + F_bar_mi)

                #nua + pi^- <-> a^- + pi^0 (t-channel)

                xj = xpic
                xk = xa
                xl = xpi0

                zeta_j = zeta_pic
                zeta_k = zeta_a
                zeta_l = 0

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vud**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + K^- <-> a^- + K^0_bar (t-channel)

                xj = xKc
                xk = xa
                xl = xK0

                zeta_j = zeta_pic
                zeta_k = zeta_a
                zeta_l = 0

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vud**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + pi^- <-> a^- + K^0 (t-channel)

                xj = xpic
                xk = xa
                xl = xK0

                zeta_j = zeta_pic
                zeta_k = zeta_a
                zeta_l = 0

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + K^- <-> a^- + pi^0 (t-channel)

                xj = xKc
                xk = xa
                xl = xpi0

                zeta_j = zeta_pic
                zeta_k = zeta_a
                zeta_l = 0

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + K^- <-> a^- + eta (t-channel)

                xj = xKc
                xk = xa
                xl = xeta

                zeta_j = zeta_pic
                zeta_k = zeta_a
                zeta_l = 0

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l_bar)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + pi^0 <-> a^- + pi^+ (t-channel)

                xj = xpi0
                xk = xa
                xl = xpic

                zeta_j = 0
                zeta_k = zeta_a
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l_bar)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vud**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + K^0 <-> a^- + K^+ (t-channel)

                xj = xK0
                xk = xa
                xl = xKc

                zeta_j = 0
                zeta_k = zeta_a
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l_bar)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vud**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + K^0_bar <-> a^- + pi^+ (t-channel)

                xj = xK0
                xk = xa
                xl = xpic

                zeta_j = 0
                zeta_k = zeta_a
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l_bar)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + pi^0 <-> a^- + K^+ (t-channel)

                xj = xpi0
                xk = xa
                xl = xKc

                zeta_j = 0
                zeta_k = zeta_a
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l_bar)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

                #nua + eta <-> a^- + K^+ (t-channel)

                xj = xeta
                xk = xa
                xl = xpic

                zeta_j = 0
                zeta_k = zeta_a
                zeta_l = zeta_pic

                Ei = i
                Ej = (j**2 + xj**2)**(1/2)
                Ek = (k**2 + xk**2)**(1/2)
                El = i + Ej - Ek

                Q = xj**2 + xk**2 - xl**2

                if (El > xl):

                    l = (El**2-xl**2)**(1/2)

                    f_j = 1/(np.exp(Ej - zeta_j)-1) 
                    f_j_bar = 1/(np.exp(Ej + zeta_j)-1) 
                    f_k = 1/(np.exp(Ek - zeta_k)+1) 
                    f_k_bar = 1/(np.exp(Ek + zeta_k)+1) 
                    f_l = 1/(np.exp(El - zeta_l)-1) 
                    f_l_bar = 1/(np.exp(El + zeta_l)-1) 

                    F = f_j*(1+f_l_bar)*(1-f_k)
                    F_bar = f_j_bar*(1+f_l)*(1-f_k_bar)

                    gam = Ei*Ej - Ei*Ek - Ej*Ek
                    kap = i**2 + k**2

                    Coll_tmp = 0

                    for ny in range(n1):

                        y = -1 + 2/(n1-1)*(ny-1)

                        eps = i*k*y
                        
                        A = j**2*(-4*kap + 8*eps)
                        B = j*(i-eps/i)*(8*gam+4*Q+8*eps)
                        C = -4*gam**2 - 4*gam*Q - Q**2 - 8*gam*eps - 4*Q*eps - 4*eps**2+4*j**2*k**2*(1-y**2)

                        if (B**2-4*A*C>0):

                            Alp=(-B+(B**2-4*A*C)**(1/2))/(2*A)
                            Bet=(-B-(B**2-4*A*C)**(1/2))/(2*A)

                            if (Bet<1) and (Bet>-1):

                                if (Alp<1) and (Alp>-1):

                                    #p1*p3

                                    pik = Ei*Ek - i*k*y

                                    #(p3*p4)(p1*p4)

                                    p_kl_il_x2 = i**2*j**2

                                    p_kl_il_x = -i*j*(2*Ei*Ej - xk**2 +Q/2 - pik)

                                    p_kl_il_0 = (Ei*Ej - xk**2 + Q/2)*(Ei*Ej - pik)
                                    
                                    
                                    #(p2*p4)(p1*p3)

                                    p_jl_ik_0 = (pik + xj**2 -Q/2)*pik

                                    #(p2*p3)(p1*p4)

                                    p_jk_il_x2 = i**2*j**2

                                    p_jk_il_x = -i*j*(2*(Ei*Ej-pik) + Q/2)

                                    p_jk_il_0 = (Ei*Ej - pik + Q/2)*(Ei*Ej - pik)

                                    #(p3*p2)(p1*p2)

                                    p_jk_ij_x2 = i**2*j**2

                                    p_jk_ij_x = -i*j*(2*Ei*Ej - pik + Q/2)

                                    p_jk_ij_0 = Ei*Ej*(Ei*Ej - pik + Q/2)
                                    

                                    #(p1*p2)(p3*p4)

                                    p_ij_kl_x2 = i**2*j**2

                                    p_ij_kl_x = -(2*Ei*Ej - xk**2 + Q/2)

                                    p_ij_kl_0 = Ei*Ej*(Ei*Ej - xk**2 + Q/2)

                                    #(p1*p3)*(m*2*2 + m4**2)

                                    pik_0 = pik*(xj**2 + xl**2)

                                    fac = 2*np.pi/(-A)**(1/2)*coe_simps[ny]*2/(n1-1)

                                    Coll_tmp += fac*(1/8*(3*Alp**2 + 2*Alp*Bet + 3*Bet**2)*(p_kl_il_x2 + p_jk_il_x2 + p_jk_ij_x2 + p_ij_kl_x2) \
                                                 + 1/2*(Alp + Bet)*(p_kl_il_x + p_jk_il_x + p_jk_ij_x + p_ij_kl_x) \
                                                 + (p_kl_il_0 - p_jl_ik_0 + p_jk_il_0 + p_jk_ij_0 + p_ij_kl_0 -1/2*pik_0))
                                    
                    
                    overall_fac = 8*Vus**2*1/(Ei*Ej*Ek)*j**2*k**2*weight_simps 

                    Coll += overall_fac*Coll_tmp*F
                    Coll_bar += overall_fac*Coll_tmp*F_bar

    Coll *= T**5*GF**2*1/(4*(2*np.pi)**4)
    Coll_bar *= T**5*GF**2*1/(4*(2*np.pi)**4)


    #2-body fusion/decay

    if (T<Tcut-10):

        if (flavor == 'e') or (flavor == 'mu'):

            #nua + a^+ <-> pi+

            xA = xpic
            zeta_A = zeta_mesonc

            Emin = ((1- xa**2/xA**2)**2*xA**2 + 4*i**2)/(4*i*(1-xa**2/xA**2))

            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua
            
        
            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))


            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
            
                    

            Coll += xA*Gamma_pi/((1-xa**2/xA**2)*i**2)*Phi 
            Coll_bar += xA*Gamma_pi/((1-xa**2/xA**2)*i**2)*Phi_bar 

            #nua + a^+ <-> K+

            xA = xKc
            zeta_A = zeta_mesonc

            Emin = ((1- xa**2/xA**2)**2*xA**2 + 4*i**2)/(4*i*(1-xa**2/xA**2))
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua


            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
            
                

            Coll += xA*Gamma_K/((1-xa**2/xA**2)*i**2)*Phi 
            Coll_bar += xA*Gamma_K/((1-xa**2/xA**2)*i**2)*Phi_bar 

            #nua + nua_bar <-> rho0

            xA = xrho0
            zeta_A = zeta_meson0

            Emin = (xA**2 + 4*i**2)/(4*i)
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua



            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
            
            
            Coll += 3*xA*Gamma_rho0/(i**2)*Phi 
            Coll_bar += 3*xA*Gamma_rho0/(i**2)*Phi_bar 

            #nua + a^+ <-> rho+

            xA = xrhoc
            zeta_A = zeta_mesonc

            Emin = ((1- xa**2/xA**2)**2*xA**2 + 4*i**2)/(4*i*(1-xa**2/xA**2))
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua
            

            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
            
            
            Coll += 3*xA*Gamma_rhoc/((1-xa**2/xA**2)*i**2)*Phi 
            Coll_bar += 3*xA*Gamma_rhoc/((1-xa**2/xA**2)*i**2)*Phi_bar 

            #nua + a^+ <-> K892+

            xA = xKc892
            zeta_A = zeta_mesonc

            Emin = ((1- xa**2/xA**2)**2*xA**2 + 4*i**2)/(4*i*(1-xa**2/xA**2))
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua
            


            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
            
                   

            Coll += 3*xA*Gamma_K892/((1-xa**2/xA**2)*i**2)*Phi 
            Coll_bar += 3*xA*Gamma_K892/((1-xa**2/xA**2)*i**2)*Phi_bar 

            #nua + nua_bar <-> omega782

            xA = xomega
            zeta_A = zeta_meson0

            Emin = (xA**2 + 4*i**2)/(4*i)
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua



            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
            
               

            Coll += 3*xA*Gamma_omega/(i**2)*Phi 
            Coll_bar += 3*xA*Gamma_omega/(i**2)*Phi_bar 

        else:

            #nua + nua <-> rho0

            xA = xrho0
            zeta_A = zeta_meson0

            Emin = (xA**2 + 4*i**2)/(4*i)
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua


            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
                        

            Coll += 3*xA*Gamma_rho0/(i**2)*Phi 
            Coll_bar += 3*xA*Gamma_rho0/(i**2)*Phi_bar 

            #nua + nua_bar <-> omega782
            

            xA = xomega
            zeta_A = zeta_meson0

            Emin = (xA**2 + 4*i**2)/(4*i)
            
            A = Emin + zeta_A 
            B = Emin + zeta_A - i + zeta_nua
            A_bar = Emin - zeta_A
            B_bar = Emin - zeta_A - i - zeta_nua



            Phi = 1/(np.exp(B-A) + 1)*np.log((1+np.exp(-B))/(1-np.exp(-A)))
            Phi_bar = 1/(np.exp(B_bar-A_bar) + 1)*np.log((1+np.exp(-B_bar))/(1-np.exp(-A_bar)))
                    

            Coll += 3*xA*Gamma_omega/(i**2)*Phi 
            Coll_bar += 3*xA*Gamma_omega/(i**2)*Phi_bar 





                
    return Coll, Coll_bar
