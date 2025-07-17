"""
This file implements the neutrino interaction rates with quarks for a fixed momentum, i.
"""

import numpy as np
from numba import jit
from Interaction_rate.D_function import D_function
from Constants import *
from Momentum_Grid import *


@jit(nopython=True,nogil=True,fastmath=True) 
def Interaction_rate_quark(T,i,zeta_nua,zeta_a,zeta_U,zeta_D,flavor):

    Coll = 0

    Coll_bar = 0

    xe = me/T
    xmu = mmu/T
    xtau = mtau/T

    Vmatrix = [[Vud,Vus],[Vcd,Vcs]]


    if (flavor == 'e'): #0: sterile nu mixing with nue, 1: sterile nu mixing with numu, 2:sterile nu mixing with nutau

        xa = xe
        

    elif (flavor == 'mu'):

        xa = xmu



    else:

        xa = xtau



    #Collision terms      
    for nk in range(n1): #bin for an integration in the collision term

        k = y_min1 + dy1*nk
        
        for nj in range(n1): #bin for an integration in the collision term
            
            weight_simps=coe_simps[nk]*coe_simps[nj]
            
            j = y_min1 + dy1*nj 

            #quark contribution above T>Lamda_QCD

            if (T>LambdaQCD + 10):

                #up-quarkとdown-quarkのflavorでsumをとる

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

                        
                        #4-point scattering

                        #nua + a^+ <-> U + D^+

                        Ej = (j**2 + xa**2)**(1/2) 
                        Ek = (k**2 + xU**2)**(1/2)
                        El = i+Ej-Ek

                        if (El > xD):

                            l = (El**2-xD**2)**(1/2)

                            fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                            fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                            fU_k = 1/(np.exp(Ek - zeta_U)+1) 
                            fU_k_bar = 1/(np.exp(Ek + zeta_U)+1) 
                            fD_l = 1/(np.exp(El - zeta_D)+1) 
                            fD_l_bar = 1/(np.exp(El + zeta_D)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            overall_fac = V**2*weight_simps*j*k*El

                            Coll += overall_fac  \
                                            *fc_j_bar*(1-fU_k)*(1 - fD_l_bar)*4*Pi_14_23   

                            Coll_bar += overall_fac  \
                                        *fc_j*(1-fU_k_bar)*(1 - fD_l)*4*Pi_14_23   
                            
                        
                        #nua + D <-> a^- + U

                        Ej = (j**2 + xD**2)**(1/2) 
                        Ek = (k**2 + xa**2)**(1/2)
                        El = i+Ej-Ek

                        if (El > xU):

                            l = (El**2-xU**2)**(1/2)

                            fD_j = 1/(np.exp(Ej - zeta_D)+1) 
                            fD_j_bar = 1/(np.exp(Ej + zeta_D)+1) 
                            fc_k = 1/(np.exp(Ek - zeta_a)+1) 
                            fc_k_bar = 1/(np.exp(Ek + zeta_a)+1) 
                            fU_l = 1/(np.exp(El - zeta_U)+1) 
                            fU_l_bar = 1/(np.exp(El + zeta_U)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_12_34   = D1 + D2_12/(i*Ej) + D2_34/(Ek*El) + D3/(i*Ej*Ek*El)

                            overall_fac = V**2*weight_simps*j*k*El

                            Coll += overall_fac  \
                                            *fD_j*(1-fc_k)*(1 - fU_l)*4*Pi_12_34   

                            Coll_bar += overall_fac  \
                                        *fD_j_bar*(1-fc_k_bar)*(1 - fU_l_bar)*4*Pi_12_34
                            

                        #nua + U^+ <-> a^- + D^+

                        Ej = (j**2 + xU**2)**(1/2) 
                        Ek = (k**2 + xa**2)**(1/2)
                        El = i+Ej-Ek

                        if (El > xD):

                            l = (El**2-xD**2)**(1/2)

                            fU_j = 1/(np.exp(Ej - zeta_U)+1) 
                            fU_j_bar = 1/(np.exp(Ej + zeta_U)+1) 
                            fc_k = 1/(np.exp(Ek - zeta_a)+1) 
                            fc_k_bar = 1/(np.exp(Ek + zeta_a)+1) 
                            fD_l = 1/(np.exp(El - zeta_D)+1) 
                            fD_l_bar = 1/(np.exp(El + zeta_D)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            overall_fac = V**2*weight_simps*j*k*El

                            Coll += overall_fac  \
                                            *fU_j_bar*(1-fc_k)*(1 - fD_l_bar)*4*Pi_14_23   

                            Coll_bar += overall_fac  \
                                        *fU_j*(1-fc_k_bar)*(1 - fD_l)*4*Pi_14_23
                            

                        #nua + nua_bar <-> U + U^+

                        Ej = j
                        Ek = (k**2 + xU**2)**(1/2)
                        El = i+Ej-Ek

                        if (El > xU):

                            l = (El**2-xU**2)**(1/2)

                            f_nua_j = 1/(np.exp(Ej - zeta_nua)+1) 
                            f_nua_j_bar = 1/(np.exp(Ej + zeta_nua)+1) 
                            fU_k = 1/(np.exp(Ek - zeta_U)+1) 
                            fU_k_bar = 1/(np.exp(Ek + zeta_U)+1) 
                            fU_l = 1/(np.exp(El - zeta_U)+1) 
                            fU_l_bar = 1/(np.exp(El + zeta_U)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            Pi_13_24 = D1 - D2_13/(i*Ek) - D2_24/(Ej*El) + D3/(i*Ej*Ek*El)

                            Pi_12 = xU**2*(D1 + D2_12/(i*Ej))/(Ek*El)

                            overall_fac = weight_simps*j*k*El*1/2 #1/2: avoiding double count in the loops

                            Coll += overall_fac  \
                                            *f_nua_j_bar*(1-fU_k)*(1 - fU_l_bar)*1/9*(16*gR**2*Pi_13_24 + (3-4*gR)**2*Pi_14_23 + 4*gR*(4*gR-3)*Pi_12)   

                            Coll_bar += overall_fac  \
                                         *f_nua_j*(1-fU_k_bar)*(1 - fU_l)*1/9*(16*gR**2*Pi_13_24 + (3-4*gR)**2*Pi_14_23 + 4*gR*(4*gR-3)*Pi_12) 


                        #nua + U <-> nua + U

                        Ej = (j**2 + xU**2)**(1/2)
                        Ek = k
                        El = i+Ej-Ek

                        if (El > xU):

                            l = (El**2-xU**2)**(1/2)

                            f_nua_k = 1/(np.exp(Ek - zeta_nua)+1) 
                            f_nua_k_bar = 1/(np.exp(Ek + zeta_nua)+1) 
                            fU_j = 1/(np.exp(Ej - zeta_U)+1) 
                            fU_j_bar = 1/(np.exp(Ej + zeta_U)+1) 
                            fU_l = 1/(np.exp(El - zeta_U)+1) 
                            fU_l_bar = 1/(np.exp(El + zeta_U)+1)

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l) 

                            Pi_12_34   = D1 + D2_12/(i*Ej) + D2_34/(Ek*El) + D3/(i*Ej*Ek*El)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            Pi_13 = xU**2*(D1 - D2_13/(i*Ek))/(Ej*El)

                            overall_fac = weight_simps*j*k*El*1/2 #1/2: avoiding double count in the loops

                            Coll += overall_fac  \
                                            *fU_j*(1-f_nua_k)*(1 - fU_l)*1/9*(16*gR**2*Pi_14_23 + (3-4*gR)**2*Pi_12_34 - 4*gR*(4*gR-3)*Pi_13)   

                            Coll_bar += overall_fac  \
                                         *fU_j_bar*(1-f_nua_k_bar)*(1 - fU_l_bar)*1/9*(16*gR**2*Pi_14_23 + (3-4*gR)**2*Pi_12_34 - 4*gR*(4*gR-3)*Pi_13)  


                        #nua + U^+ <-> nua + U^+ 

                        Ej = (j**2 + xU**2)**(1/2)
                        Ek = k
                        El = i+Ej-Ek

                        if (El > xU):

                            l = (El**2-xU**2)**(1/2)

                            f_nua_k = 1/(np.exp(Ek - zeta_nua)+1) 
                            f_nua_k_bar = 1/(np.exp(Ek + zeta_nua)+1) 
                            fU_j = 1/(np.exp(Ej - zeta_U)+1) 
                            fU_j_bar = 1/(np.exp(Ej + zeta_U)+1) 
                            fU_l = 1/(np.exp(El - zeta_U)+1) 
                            fU_l_bar = 1/(np.exp(El + zeta_U)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_12_34   = D1 + D2_12/(i*Ej) + D2_34/(Ek*El) + D3/(i*Ej*Ek*El)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            Pi_13 = xU**2*(D1 - D2_13/(i*Ek))/(Ej*El)

                            overall_fac = weight_simps*j*k*El*1/2 #1/2: avoiding double count in the loops

                            Coll += overall_fac  \
                                            *fU_j_bar*(1-f_nua_k)*(1 - fU_l_bar)*1/9*(16*gR**2*Pi_12_34 + (3-4*gR)**2*Pi_14_23 - 4*gR*(4*gR-3)*Pi_13)   

                            Coll_bar += overall_fac  \
                                        *fU_j*(1-f_nua_k_bar)*(1 - fU_l)*1/9*(16*gR**2*Pi_12_34 + (3-4*gR)**2*Pi_14_23 - 4*gR*(4*gR-3)*Pi_13)  
                            

                        #nua + nua_bar <-> D + D^+ 

                        Ej = j
                        Ek = (k**2 + xD**2)**(1/2)
                        El = i+Ej-Ek

                        if (El > xD):

                            l = (El**2-xD**2)**(1/2)

                            f_nua_j = 1/(np.exp(Ej - zeta_nua)+1) 
                            f_nua_j_bar = 1/(np.exp(Ej + zeta_nua)+1) 
                            fD_k = 1/(np.exp(Ek - zeta_D)+1) 
                            fD_k_bar = 1/(np.exp(Ek + zeta_D)+1) 
                            fD_l = 1/(np.exp(El - zeta_D)+1) 
                            fD_l_bar = 1/(np.exp(El + zeta_D)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            Pi_13_24 = D1 - D2_13/(i*Ek) - D2_24/(Ej*El) + D3/(i*Ej*Ek*El)

                            Pi_12 = xD**2*(D1 + D2_12/(i*Ej))/(Ek*El)

                            overall_fac = weight_simps*j*k*El*1/2 #1/2: avoiding double count in the loops

                            Coll += overall_fac  \
                                            *f_nua_j_bar*(1-fD_k)*(1 - fD_l_bar)*1/9*(4*gR**2*Pi_13_24 + (3-2*gR)**2*Pi_14_23 + 2*gR*(2*gR-3)*Pi_12)   

                            Coll_bar += overall_fac  \
                                        *f_nua_j*(1-fD_k_bar)*(1 - fD_l)*1/9*(4*gR**2*Pi_13_24 + (3-2*gR)**2*Pi_14_23 + 2*gR*(2*gR-3)*Pi_12)


                        #nua + D <-> nua + D

                        Ej = (j**2 + xD**2)**(1/2)
                        Ek = k
                        El = i+Ej-Ek

                        if (El > xD):

                            l = (El**2-xD**2)**(1/2)

                            f_nua_k = 1/(np.exp(Ek - zeta_nua)+1) 
                            f_nua_k_bar = 1/(np.exp(Ek + zeta_nua)+1) 
                            fD_j = 1/(np.exp(Ej - zeta_D)+1) 
                            fD_j_bar = 1/(np.exp(Ej + zeta_D)+1) 
                            fD_l = 1/(np.exp(El - zeta_D)+1) 
                            fD_l_bar = 1/(np.exp(El + zeta_D)+1)

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l) 

                            Pi_12_34   = D1 + D2_12/(i*Ej) + D2_34/(Ek*El) + D3/(i*Ej*Ek*El)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            Pi_13 = xD**2*(D1 - D2_13/(i*Ek))/(Ej*El)

                            overall_fac = weight_simps*j*k*El*1/2 #1/2: avoiding double count in the loops

                            Coll += overall_fac  \
                                            *fD_j*(1-f_nua_k)*(1 - fD_l)*1/9*(4*gR**2*Pi_14_23 + (3-2*gR)**2*Pi_12_34 - 2*gR*(2*gR-3)*Pi_13)   

                            Coll_bar += overall_fac  \
                                         *fD_j_bar*(1-f_nua_k_bar)*(1 - fD_l_bar)*1/9*(4*gR**2*Pi_14_23 + (3-2*gR)**2*Pi_12_34 - 2*gR*(2*gR-3)*Pi_13) 
                            
                        #nua + D^+ <-> nua + D^+ 

                        Ej = (j**2 + xD**2)**(1/2)
                        Ek = k
                        El = i+Ej-Ek

                        if (El > xD):

                            l = (El**2-xD**2)**(1/2)

                            f_nua_k = 1/(np.exp(Ek - zeta_nua)+1) 
                            f_nua_k_bar = 1/(np.exp(Ek + zeta_nua)+1) 
                            fD_j = 1/(np.exp(Ej - zeta_D)+1) 
                            fD_j_bar = 1/(np.exp(Ej + zeta_D)+1) 
                            fD_l = 1/(np.exp(El - zeta_D)+1) 
                            fD_l_bar = 1/(np.exp(El + zeta_D)+1) 

                            D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                            Pi_12_34   = D1 + D2_12/(i*Ej) + D2_34/(Ek*El) + D3/(i*Ej*Ek*El)

                            Pi_14_23 = D1 - D2_23/(Ej*Ek) - D2_14/(i*El) + D3/(i*Ej*Ek*El)

                            Pi_13 = xD**2*(D1 - D2_13/(i*Ek))/(Ej*El)

                            overall_fac = weight_simps*j*k*El*1/2 #1/2: avoiding double count in the loops

                            Coll += overall_fac  \
                                            *fD_j_bar*(1-f_nua_k)*(1 - fD_l_bar)*1/9*(4*gR**2*Pi_12_34 + (3-2*gR)**2*Pi_14_23 - 2*gR*(2*gR-3)*Pi_13)   

                            Coll_bar += overall_fac  \
                                        *fD_j*(1-f_nua_k_bar)*(1 - fD_l)*1/9*(4*gR**2*Pi_12_34 + (3-2*gR)**2*Pi_14_23 - 2*gR*(2*gR-3)*Pi_13) 


                        #3-body decay

                        if (flavor == 'e'): #nue

                            if (upquark == 0): #case of up quark

                                #nue + e^+ + U^+ <-> D^+

                                Ej = (j**2 + xa**2)**(1/2)
                                Ek = (k**2 + xU**2)**(1/2)
                                El = i+Ej+Ek

                                if (El > xD):

                                    l = (El**2-xD**2)**(1/2)

                                    fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                                    fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                                    fU_k = 1/(np.exp(Ek - zeta_U)+1) 
                                    fU_k_bar = 1/(np.exp(Ek + zeta_U)+1) 
                                    fD_l = 1/(np.exp(El - zeta_D)+1) 
                                    fD_l_bar = 1/(np.exp(El + zeta_D)+1) 

                                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                                    Pi_14_23 = D1 + D2_23/(Ej*Ek) - D2_14/(i*El) - D3/(i*Ej*Ek*El)

                                    overall_fac = V**2*weight_simps*j*k*El

                                    Coll += overall_fac  \
                                                    *fc_j_bar*fU_k_bar*(1 - fD_l_bar)*4*Pi_14_23   

                                    Coll_bar += overall_fac  \
                                                *fc_j*fU_k*(1 - fD_l)*4*Pi_14_23   
                                    

                            else: #case of charm quark

                                #nue + e^+ + D^- <-> U^-
                                
                                Ej = (j**2 + xa**2)**(1/2)
                                Ek = (k**2 + xD**2)**(1/2)
                                El = i+Ej+Ek

                                if (El > xU):

                                    l = (El**2-xU**2)**(1/2)

                                    fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                                    fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                                    fD_k = 1/(np.exp(Ek - zeta_D)+1) 
                                    fD_k_bar = 1/(np.exp(Ek + zeta_D)+1) 
                                    fU_l = 1/(np.exp(El - zeta_U)+1) 
                                    fU_l_bar = 1/(np.exp(El + zeta_U)+1) 

                                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                                    Pi_13_24 = D1 + D2_13/(i*Ek) - D2_24/(Ej*El) - D3/(i*Ej*Ek*El)

                                    overall_fac = V**2*weight_simps*j*k*El

                                    Coll += overall_fac  \
                                                    *fc_j_bar*fU_k*(1 - fD_l)*4*Pi_13_24   

                                    Coll_bar += overall_fac  \
                                                *fc_j*fU_k_bar*(1 - fD_l_bar)*4*Pi_13_24   


                        elif (flavor == 'mu'): #numu

                            if (upquark == 0): #case of up quark

                                #numu + U^+ + D <-> mu^-

                                Ej = (j**2 + xU**2)**(1/2)
                                Ek = (k**2 + xD**2)**(1/2)
                                El = i+Ej+Ek

                                if (El > xa):

                                    l = (El**2-xa**2)**(1/2)

                                    fU_j = 1/(np.exp(Ej - zeta_U)+1) 
                                    fU_j_bar = 1/(np.exp(Ej + zeta_U)+1) 
                                    fD_k = 1/(np.exp(Ek - zeta_U)+1) 
                                    fD_k_bar = 1/(np.exp(Ek + zeta_D)+1) 
                                    fc_l = 1/(np.exp(El - zeta_a)+1) 
                                    fc_l_bar = 1/(np.exp(El + zeta_a)+1) 

                                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                                    Pi_13_24 = D1 + D2_13/(i*Ek) - D2_24/(Ej*El) - D3/(i*Ej*Ek*El)


                                    overall_fac = V**2*weight_simps*j*k*El

                                    Coll += overall_fac  \
                                                    *fU_j_bar*fD_k*(1 - fc_l)*4*Pi_13_24   

                                    Coll_bar += overall_fac  \
                                                *fU_j*fD_k_bar*(1 - fc_l_bar)*4*Pi_13_24 

                            else: #case of charm quark

                                #numu + mu^+ + D^- <-> U^-

                                Ej = (j**2 + xa**2)**(1/2)
                                Ek = (k**2 + xD**2)**(1/2)
                                El = i + Ej + Ek

                                if (El > xU):

                                    l = (El**2-xU**2)**(1/2)

                                    fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                                    fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                                    fD_k = 1/(np.exp(Ek - zeta_D)+1) 
                                    fD_k_bar = 1/(np.exp(Ek + zeta_D)+1) 
                                    fU_l = 1/(np.exp(El - zeta_U)+1) 
                                    fU_l_bar = 1/(np.exp(El + zeta_U)+1) 

                                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                                    Pi_13_24 = D1 + D2_13/(i*Ek) - D2_24/(Ej*El) - D3/(i*Ej*Ek*El)


                                    overall_fac = V**2*weight_simps*j*k*El

                                    Coll += overall_fac  \
                                                    *fc_j_bar*fD_k*(1 - fU_l)*4*Pi_13_24   

                                    Coll_bar += overall_fac  \
                                                *fc_j*fD_k_bar*(1 - fU_l_bar)*4*Pi_13_24   

                        else: #nutau

                            #nutau + U^+ + D <-> tau^-

                            Ej = (j**2 + xU**2)**(1/2)
                            Ek = (k**2 + xD**2)**(1/2)
                            El = i+Ej+Ek

                            if (El > xa):

                                l = (El**2-xD**2)**(1/2)

                                fU_j = 1/(np.exp(Ej - zeta_U)+1) 
                                fU_j_bar = 1/(np.exp(Ej + zeta_U)+1) 
                                fD_k = 1/(np.exp(Ek - zeta_D)+1) 
                                fD_k_bar = 1/(np.exp(Ek + zeta_D)+1) 
                                fc_l = 1/(np.exp(El - zeta_a)+1) 
                                fc_l_bar = 1/(np.exp(El + zeta_a)+1) 

                                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                                Pi_13_24 = D1 + D2_13/(i*Ek) - D2_24/(Ej*El) - D3/(i*Ej*Ek*El)

                                overall_fac = V**2*weight_simps*j*k*El

                                Coll += overall_fac  \
                                                *fU_j_bar*fD_k*(1 - fc_l)*4*Pi_13_24   

                                Coll_bar += overall_fac  \
                                            *fU_j*fD_k_bar*(1 - fc_l_bar)*4*Pi_13_24     

    Coll *= 3*T**5*GF**2/(2*np.pi**3*i) #3: d.o.f for colors
    Coll_bar *= 3*T**5*GF**2/(2*np.pi**3*i)

    return Coll, Coll_bar
