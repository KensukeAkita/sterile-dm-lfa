import numpy as np
from numba import jit
from Collision_term.D_function import D_function
from Constants import *
from Sterile_Nu_Parameters import *
from Momentum_Grid import *

@jit(nopython=True,nogil=True,fastmath=True)
def Collision_term_lepton(T,i,zeta_nue,zeta_numu,zeta_nutau,zeta_e,zeta_mu,zeta_tau):

    Coll = 0

    Coll_bar = 0

    xe = me/T
    xmu = mmu/T
    xtau = mtau/T

    if (flavor == 'e'): 

        xa = xe
        xb = xmu
        xc = xtau  #always xb < xc
        
        zeta_a = zeta_e
        zeta_b = zeta_mu
        zeta_c = zeta_tau 

        zeta_nua = zeta_nue
        zeta_nub = zeta_numu
        zeta_nuc = zeta_nutau 

        #zeta_a = 0
        #zeta_b = 0
        #zeta_c = 0 


    elif (flavor == 'mu'):

        xa = xmu
        xb = xe
        xc = xtau

        zeta_a = zeta_mu
        zeta_b = zeta_e
        zeta_c = zeta_tau

        zeta_nua = zeta_numu
        zeta_nub = zeta_nue
        zeta_nuc = zeta_nutau 


        #zeta_a = 0
        #zeta_b = 0
        #zeta_c = 0


    else:

        xa = xtau
        xb = xe
        xc = xmu
        
        zeta_a = zeta_tau
        zeta_b = zeta_e
        zeta_c = zeta_mu

        zeta_nua = zeta_nutau
        zeta_nub = zeta_nue
        zeta_nuc = zeta_numu 


        #zeta_a = 0
        #zeta_b = 0
        #zeta_c = 0




    #Collision terms      
    for nk in range(n1): #bin for an integration in the collision term

        k = y_min1 + dy1*nk

        f_nua_k = 1/(np.exp(k-zeta_nua)+1)
        f_nua_bar_k = 1/(np.exp(k+zeta_nua)+1)
        f_nub_k = 1/(np.exp(k-zeta_nub)+1)
        f_nub_bar_k = 1/(np.exp(k+zeta_nub)+1)
        f_nuc_k = 1/(np.exp(k-zeta_nuc)+1)
        f_nuc_bar_k = 1/(np.exp(k+zeta_nuc)+1)



        
        for nj in range(n1): #bin for an integration in the collision term
            
            weight_simps=coe_simps[nk]*coe_simps[nj]
            
            j = y_min1 + dy1*nj 

            f_nua_j = 1/(np.exp(j-zeta_nua)+1)
            f_nua_bar_j = 1/(np.exp(j+zeta_nua)+1)
            f_nub_j = 1/(np.exp(j-zeta_nub)+1)
            f_nub_bar_j = 1/(np.exp(j+zeta_nub)+1)
            f_nuc_j = 1/(np.exp(j-zeta_nuc)+1)
            f_nuc_bar_j = 1/(np.exp(j+zeta_nuc)+1)

            #neutrino self-interations

            l = i + j - k
            
            if (l >= 0):

                f_nua_l = 1/(np.exp(l-zeta_nua)+1)
                f_nua_bar_l = 1/(np.exp(l+zeta_nua)+1)
                f_nub_l = 1/(np.exp(l-zeta_nub)+1)
                f_nub_bar_l = 1/(np.exp(l+zeta_nub)+1)
                f_nuc_l = 1/(np.exp(l-zeta_nuc)+1)
                f_nuc_bar_l = 1/(np.exp(l+zeta_nuc)+1)
                    
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)


                Pi_self_12_34 = D1 + D2_12/(i*j) + D2_34/(k*l) + D3/(i*j*k*l)
                Pi_self_14_23 = D1 - D2_14/(i*l) - D2_23/(j*k) + D3/(i*j*k*l)

                
                overall_fac_self = j*k*l*weight_simps
            
                
                Coll += overall_fac_self \
                    *((1 - f_nua_k)*f_nua_bar_j*(1 - f_nua_bar_l)*4*Pi_self_14_23 \
                    +(1 - f_nua_k)*f_nua_j*(1 - f_nua_l)*2*Pi_self_12_34 \
                    +(1 - f_nua_k)*f_nub_bar_j*(1 - f_nub_bar_l)*Pi_self_14_23 \
                    +(1 - f_nua_k)*f_nub_j*(1 - f_nub_l)*Pi_self_12_34 \
                    +f_nua_bar_j*(1-f_nub_k)*(1 - f_nub_bar_l)*Pi_self_14_23 \
                    +(1 - f_nua_k)*f_nuc_bar_j*(1 - f_nuc_bar_l)*Pi_self_14_23 \
                    +(1 - f_nua_k)*f_nuc_j*(1 - f_nuc_l)*Pi_self_12_34 \
                    +f_nua_bar_j*(1-f_nuc_k)*(1 - f_nuc_bar_l)*Pi_self_14_23)


                #anti neutrino self-interaction
                
                Coll_bar += overall_fac_self \
                    *((1 - f_nua_bar_k)*f_nua_j*(1 - f_nua_l)*4*Pi_self_14_23 \
                    +(1 - f_nua_bar_k)*f_nua_bar_j*(1 - f_nua_bar_l)*2*Pi_self_12_34 \
                    +(1 - f_nua_bar_k)*f_nub_j*(1 - f_nub_l)*Pi_self_14_23 \
                    +(1 - f_nua_bar_k)*f_nub_bar_j*(1 - f_nub_bar_l)*Pi_self_12_34 \
                    +f_nua_j*(1-f_nub_bar_k)*(1 - f_nub_l)*Pi_self_14_23 \
                    +(1 - f_nua_bar_k)*f_nuc_j*(1 - f_nuc_l)*Pi_self_14_23 \
                    +(1 - f_nua_bar_k)*f_nuc_bar_j*(1 - f_nuc_bar_l)*Pi_self_12_34 \
                    +f_nua_j*(1-f_nuc_bar_k)*(1 - f_nuc_l)*Pi_self_14_23)

                
            #nu c^+- <-> nu c^+- (c:charged lepton) 
           
            #nu a^+- <-> nu a^+-   

            Ej = (j**2 + xa**2)**(1/2) #Ej=E/T
            El = i+Ej-k
                
            if (El > xa):
                
                l = (El**2-xa**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                fc_j_bar = 1/(np.exp(Ej + zeta_a)+1)
                fc_l = 1/(np.exp(El - zeta_a)+1)
                fc_l_bar = 1/(np.exp(El + zeta_a)+1)
                    
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_12_34 = D1 + D2_12/(i*Ej) + D2_34/(k*El) + D3/(i*Ej*k*El)
                Pi_sc_14_23 = D1 - D2_23/(Ej*k) - D2_14/(i*El) + D3/(i*Ej*k*El)
                Pi_sc_13 = xa**2*(D1 - D2_13/(i*k))/(Ej*El)

                overall_fac_sc = weight_simps*j*k*El

                #nu a^- <-> nu a^-    
                    
                Coll += overall_fac_sc  \
                                *fc_j*(1 - fc_l)*(1-f_nua_k)*(Pi_sc_12_34*4*gL**2 + Pi_sc_14_23*4*gR**2 - Pi_sc_13*4*gL*gR)            
                

                #nu a^+ <-> nu a^+   
                
                Coll += overall_fac_sc  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nua_k)*(Pi_sc_12_34*4*gR**2 + Pi_sc_14_23*4*gL**2 - Pi_sc_13*4*gL*gR) 
                
                
                #nubar a^+ <-> nubar a^+    
                    
                Coll_bar += overall_fac_sc  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nua_bar_k)*(Pi_sc_12_34*4*gL**2 + Pi_sc_14_23*4*gR**2 - Pi_sc_13*4*gL*gR)            
                
                
                #nubar a^- <-> nubar a^-   
                
                Coll_bar += overall_fac_sc  \
                                *fc_j*(1 - fc_l)*(1-f_nua_bar_k)*(Pi_sc_12_34*4*gR**2 + Pi_sc_14_23*4*gL**2 - Pi_sc_13*4*gL*gR) 
                

       
            #nu b^+- <-> nu b^+-    

            Ej = (j**2 + xb**2)**(1/2) #Ej=E/T
            El = i+Ej-k
                
            if (El > xb):
                
                l = (El**2-xb**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_b)+1)
                fc_j_bar = 1/(np.exp(Ej + zeta_b)+1)
                fc_l = 1/(np.exp(El - zeta_b)+1)
                fc_l_bar = 1/(np.exp(El + zeta_b)+1)
                    
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_12_34 = D1 + D2_12/(i*Ej) + D2_34/(k*El) + D3/(i*Ej*k*El)
                Pi_sc_14_23 = D1 - D2_23/(Ej*k) - D2_14/(i*El) + D3/(i*Ej*k*El)
                Pi_sc_13 = xb**2*(D1 - D2_13/(i*k))/(Ej*El)

                overall_fac_sc = weight_simps*j*k*El

                #nu b^- <-> nu b^-    
                    
                Coll += overall_fac_sc  \
                                *fc_j*(1 - fc_l)*(1-f_nua_k)*(Pi_sc_12_34*4*gLtilde**2 + Pi_sc_14_23*4*gR**2 - Pi_sc_13*4*gLtilde*gR)            
                
                
                #nu b^+ <-> nu b^+   
                
                Coll += overall_fac_sc  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nua_k)*(Pi_sc_12_34*4*gR**2 + Pi_sc_14_23*4*gLtilde**2 - Pi_sc_13*4*gLtilde*gR) 
                
                
                #nubar b^+ <-> nubar b^+    
                    
                Coll_bar += overall_fac_sc  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nua_bar_k)*(Pi_sc_12_34*4*gLtilde**2 + Pi_sc_14_23*4*gR**2 - Pi_sc_13*4*gLtilde*gR)            
                
                
                #nubar b^- <-> nubar b^-   
                
                Coll_bar += overall_fac_sc  \
                                *fc_j*(1 - fc_l)*(1-f_nua_bar_k)*(Pi_sc_12_34*4*gR**2 + Pi_sc_14_23*4*gLtilde**2 - Pi_sc_13*4*gLtilde*gR) 
                
                

            #nu c^+- <-> nu c^+-    

            Ej = (j**2 + xc**2)**(1/2) #Ej=E/T
            El = i+Ej-k
                
            if (El > xc):
                
                l = (El**2-xc**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_c)+1) 
                fc_j_bar = 1/(np.exp(Ej + zeta_c)+1) 
                fc_l = 1/(np.exp(El - zeta_c)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_c)+1) 
                    
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_12_34 = D1 + D2_12/(i*Ej) + D2_34/(k*El) + D3/(i*Ej*k*El)
                Pi_sc_14_23 = D1 - D2_23/(Ej*k) - D2_14/(i*El) + D3/(i*Ej*k*El)
                Pi_sc_13 = xc**2*(D1 - D2_13/(i*k))/(Ej*El)

                overall_fac_sc = weight_simps*j*k*El

                #nu c^- <-> nu c^-    
                    
                Coll += overall_fac_sc  \
                                *fc_j*(1 - fc_l)*(1-f_nua_k)*(Pi_sc_12_34*4*gLtilde**2 + Pi_sc_14_23*4*gR**2 - Pi_sc_13*4*gLtilde*gR)            
                
                
                #nubar c^+ <-> nubar c^+   
                
                Coll += overall_fac_sc  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nua_k)*(Pi_sc_12_34*4*gR**2 + Pi_sc_14_23*4*gLtilde**2 - Pi_sc_13*4*gLtilde*gR) 
                
            
                #nubar c^+ <-> nubar c^+    
                    
                Coll_bar += overall_fac_sc  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nua_bar_k)*(Pi_sc_12_34*4*gLtilde**2 + Pi_sc_14_23*4*gR**2 - Pi_sc_13*4*gLtilde*gR)            
                
                
                #nubar c^- <-> nubar c^-   
                
                Coll_bar += overall_fac_sc  \
                                *fc_j*(1 - fc_l)*(1-f_nua_bar_k)*(Pi_sc_12_34*4*gR**2 + Pi_sc_14_23*4*gLtilde**2 - Pi_sc_13*4*gLtilde*gR) 
                

            
            
            #nu nubar <-> c^- c^+ (c:charged lepton)

            #nu nubar <-> a^- a^+
                
            Ek = (k**2 + xa**2)**(1/2)
            El = i + j - Ek
            
            if (El > xa):
                
                l = (El**2 - xa**2)**(1/2)

                fc_k = 1/(np.exp(El - zeta_a)+1) 
                fc_k_bar = 1/(np.exp(El + zeta_a)+1) 
                fc_l = 1/(np.exp(El - zeta_a)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_a)+1) 
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                
                Pi_ann_14_23 = D1 - D2_23/(j*Ek) - D2_14/(i*El) + D3/(i*j*Ek*El)
                
                Pi_ann_13_24 = D1 -  D2_24/(j*El) - D2_13/(i*Ek) + D3/(i*j*Ek*El)
                
                Pi_ann_12 = xa**2*(D1 + D2_12/(i*j))/(Ek*El)

                overall_fac_ann = weight_simps*j*k*El

                #nu nubar <-> a^- a^+
                               
                Coll += overall_fac_ann  \
                            *(1 - fc_k)*(1 - fc_l_bar)*f_nua_bar_j*(Pi_ann_14_23*4*gL**2+ Pi_ann_13_24*4*gR**2 + Pi_ann_12*4*gL*gR)           
                
                
                #nubar nu <-> a^+ a^-
                               
                Coll_bar += overall_fac_ann  \
                            *(1 - fc_k_bar)*(1 - fc_l)*f_nua_j*(Pi_ann_14_23*4*gL**2+ Pi_ann_13_24*4*gR**2 + Pi_ann_12*4*gL*gR)           
                
            
            #nu nubar <-> b^- b^+
                
            Ek = (k**2 + xb**2)**(1/2)
            El = i + j - Ek
            
            if (El > xb):
                
                l = (El**2 - xb**2)**(1/2)

                fc_k = 1/(np.exp(El - zeta_b)+1) 
                fc_k_bar = 1/(np.exp(El + zeta_b)+1) 
                fc_l = 1/(np.exp(El - zeta_b)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_b)+1) 
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)
                
                Pi_ann_14_23 = D1 - D2_23/(j*Ek) - D2_14/(i*El) + D3/(i*j*Ek*El)
                
                Pi_ann_13_24 = D1 -  D2_24/(j*El) - D2_13/(i*Ek) + D3/(i*j*Ek*El)
                
                Pi_ann_12 = xb**2*(D1 + D2_12/(i*j))/(Ek*El)

                overall_fac_ann = weight_simps*j*k*El

                #nu nubar <-> b^- b^+
                               
                Coll += overall_fac_ann  \
                            *(1 - fc_k)*(1 - fc_l_bar)*f_nua_bar_j*(Pi_ann_14_23*4*gLtilde**2+ Pi_ann_13_24*4*gR**2 + Pi_ann_12*4*gLtilde*gR)           
                
                
                #nubar nu <-> b^+ b^-
                               
                Coll_bar += overall_fac_ann  \
                            *(1 - fc_k_bar)*(1 - fc_l)*f_nua_j*(Pi_ann_14_23*4*gLtilde**2+ Pi_ann_13_24*4*gR**2 + Pi_ann_12*4*gLtilde*gR)           
                
                

            #nu nubar <-> c^- c^+
                
            Ek = (k**2 + xc**2)**(1/2)
            El = i + j - Ek
            
            if (El > xc):
                
                l = (El**2 - xc**2)**(1/2)

                fc_k = 1/(np.exp(El - zeta_c)+1) 
                fc_k_bar = 1/(np.exp(El + zeta_c)+1) 
                fc_l = 1/(np.exp(El - zeta_c)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_c)+1) 
                
                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)
                
                Pi_ann_14_23 = D1 - D2_23/(j*Ek) - D2_14/(i*El) + D3/(i*j*Ek*El)
                
                Pi_ann_13_24 = D1 -  D2_24/(j*El) - D2_13/(i*Ek) + D3/(i*j*Ek*El)
                
                Pi_ann_12 = xc**2*(D1 + D2_12/(i*j))/(Ek*El)

                overall_fac_ann = weight_simps*j*k*El

                #nu nubar <-> c^- c^+
                               
                Coll += overall_fac_ann  \
                            *(1 - fc_k)*(1 - fc_l_bar)*f_nua_bar_j*(Pi_ann_14_23*4*gLtilde**2+ Pi_ann_13_24*4*gR**2 + Pi_ann_12*4*gLtilde*gR)           

                
                #nubar nu <-> c^+ c^-
                               
                Coll_bar += overall_fac_ann  \
                            *(1 - fc_k_bar)*(1 - fc_l)*f_nua_j*(Pi_ann_14_23*4*gLtilde**2+ Pi_ann_13_24*4*gR**2 + Pi_ann_12*4*gLtilde*gR)           
                

            #Mixed neutrino flavor reactions for electron neutrinos

            #nua a^+ <-> nub b^+

            Ej = (j**2 + xa**2)**(1/2)
            El = i + Ej - k

            if (El > xb):

                l = (El**2-xb**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                fc_l = 1/(np.exp(El - zeta_b)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_b)+1) 

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_14_23 = D1 - D2_23/(Ej*k) - D2_14/(i*El) + D3/(i*Ej*k*El)

                overall_fac = weight_simps*j*k*El

                Coll += overall_fac  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nub_k)*4*Pi_sc_14_23   

                Coll_bar += overall_fac  \
                            *fc_j*(1 - fc_l)*(1-f_nub_bar_k)*4*Pi_sc_14_23 


            #nua a^+ <-> nuc c^+

            Ej = (j**2 + xa**2)**(1/2)
            El = i + Ej - k

            if (El > xc):

                l = (El**2-xc**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                fc_l = 1/(np.exp(El - zeta_c)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_c)+1) 

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_14_23 = D1 - D2_23/(Ej*k) - D2_14/(i*El) + D3/(i*Ej*k*El)

                overall_fac = weight_simps*j*k*El

                Coll += overall_fac  \
                                *fc_j_bar*(1 - fc_l_bar)*(1-f_nuc_k)*4*Pi_sc_14_23   

                Coll_bar += overall_fac  \
                            *fc_j*(1 - fc_l)*(1-f_nuc_bar_k)*4*Pi_sc_14_23 
                
                
            #nua b^- <-> nub a^- 

            Ej = (j**2 + xb**2)**(1/2)
            El = i + Ej - k

            if (El > xa):

                l = (El**2-xa**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_b)+1) 
                fc_j_bar = 1/(np.exp(Ej + zeta_b)+1) 
                fc_l = 1/(np.exp(El - zeta_a)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_a)+1) 

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_12_34   = D1 + D2_12/(i*Ej) + D2_34/(k*El) + D3/(i*Ej*k*El)

                overall_fac = weight_simps*j*k*El

                Coll += overall_fac  \
                                *fc_j*(1 - fc_l)*(1-f_nub_k)*4*Pi_sc_12_34   

                Coll_bar += overall_fac  \
                            *fc_j_bar*(1 - fc_l_bar)*(1-f_nub_k)*4*Pi_sc_12_34 



            #nua c^- <-> nuc a^- 

            Ej = (j**2 + xc**2)**(1/2)
            El = i + Ej - k

            if (El > xa):

                l = (El**2-xa**2)**(1/2)

                fc_j = 1/(np.exp(Ej - zeta_c)+1) 
                fc_j_bar = 1/(np.exp(Ej + zeta_c)+1) 
                fc_l = 1/(np.exp(El - zeta_a)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_a)+1) 

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_sc_12_34   = D1 + D2_12/(i*Ej) + D2_34/(k*El) + D3/(i*Ej*k*El)

                overall_fac = weight_simps*j*k*El

                Coll += overall_fac  \
                                *fc_j*(1 - fc_l)*(1-f_nuc_k)*4*Pi_sc_12_34   

                Coll_bar += overall_fac  \
                            *fc_j_bar*(1 - fc_l_bar)*(1-f_nuc_k)*4*Pi_sc_12_34 


            #nua nubbar <-> a^- b^+

            Ek = (k**2 + xa**2)**(1/2)
            El = i + j - Ek

            if (El > xb):

                l = (El**2-xb**2)**(1/2)

                fc_k = 1/(np.exp(Ek - zeta_a)+1) 
                fc_k_bar = 1/(np.exp(Ek + zeta_a)+1) 
                fc_l = 1/(np.exp(El - zeta_b)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_b)+1)

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_ann_14_23 = D1 - D2_23/(j*Ek) - D2_14/(i*El) + D3/(i*j*Ek*El)

                overall_fac = weight_simps*j*k*El

                Coll += overall_fac  \
                                *(1-fc_k)*(1 - fc_l_bar)*f_nub_bar_j*4*Pi_ann_14_23   

                Coll_bar += overall_fac  \
                                *(1-fc_k_bar)*(1 - fc_l)*f_nub_j*4*Pi_ann_14_23 



            #nua nucbar <-> a^- c^+

            Ek = (k**2 + xa**2)**(1/2)
            El = i + j - Ek

            if (El > xc):

                l = (El**2-xc**2)**(1/2)

                fc_k = 1/(np.exp(Ek - zeta_a)+1) 
                fc_k_bar = 1/(np.exp(Ek + zeta_a)+1) 
                fc_l = 1/(np.exp(El - zeta_c)+1) 
                fc_l_bar = 1/(np.exp(El + zeta_c)+1)

                D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                Pi_ann_14_23 = D1 - D2_23/(j*Ek) - D2_14/(i*El) + D3/(i*j*Ek*El)

                overall_fac = weight_simps*j*k*El

                Coll += overall_fac  \
                                *(1-fc_k)*(1 - fc_l_bar)*f_nub_bar_j*4*Pi_ann_14_23   

                Coll_bar += overall_fac  \
                                *(1-fc_k_bar)*(1 - fc_l)*f_nub_j*4*Pi_ann_14_23 
                

            #3-body decay
            #Note that the sign in front of the D-functions changes, compared to the 2-to-2 processes.

            if (flavor == 'e'):

                #nue e^+ numubar <-> mu^+
                #nua a^+ nubbar <-> b^+

                Ej = (j**2 + xa**2)**(1/2)
                El = i + Ej + k

                if (El > xb):

                    l = (El**2-xb**2)**(1/2)

                    fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                    fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                    fc_l = 1/(np.exp(El - zeta_b)+1) 
                    fc_l_bar = 1/(np.exp(El + zeta_b)+1)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    Pi_sc_14_23 = D1 + D2_23/(Ej*k) - D2_14/(i*El) - D3/(i*Ej*k*El)

                    overall_fac = weight_simps*j*k*El

                    Coll += overall_fac  \
                                *(1-fc_l_bar)*fc_j_bar*f_nub_bar_k*4*Pi_sc_14_23   

                    Coll_bar += overall_fac  \
                                *(1-fc_l)*fc_j*f_nub_k*4*Pi_sc_14_23 


                #nue e^+ nutaubar <-> tau^+
                #nua a^+ nucbar <-> c^+

                if (El > xc):

                    l = (El**2-xc**2)**(1/2)

                    fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                    fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                    fc_l = 1/(np.exp(El - zeta_c)+1) 
                    fc_l_bar = 1/(np.exp(El + zeta_c)+1)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    Pi_sc_14_23 = D1 + D2_23/(Ej*k) - D2_14/(i*El) - D3/(i*Ej*k*El)

                    overall_fac = weight_simps*j*k*El

                    Coll += overall_fac  \
                                *(1-fc_l_bar)*fc_j_bar*f_nuc_bar_k*4*Pi_sc_14_23   

                    Coll_bar += overall_fac  \
                                *(1-fc_l)*fc_j*f_nuc_k*4*Pi_sc_14_23 



            elif (flavor == 'mu'):

                #numu mu^+ nutaubar <-> tau^+
                #nua a^+ nucbar <-> c^+

                Ej = (j**2 + xa**2)**(1/2)
                El = i + Ej + k

                if (El > xc):

                    l = (El**2-xc**2)**(1/2)

                    fc_j = 1/(np.exp(Ej - zeta_a)+1) 
                    fc_j_bar = 1/(np.exp(Ej + zeta_a)+1) 
                    fc_l = 1/(np.exp(El - zeta_c)+1) 
                    fc_l_bar = 1/(np.exp(El + zeta_c)+1)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    Pi_sc_14_23 = D1 + D2_23/(Ej*k) - D2_14/(i*El) - D3/(i*Ej*k*El)

                    overall_fac = weight_simps*j*k*El

                    Coll += overall_fac  \
                                *(1-fc_l_bar)*fc_j_bar*f_nuc_bar_k*4*Pi_sc_14_23   

                    Coll_bar += overall_fac  \
                                *(1-fc_l)*fc_j*f_nuc_k*4*Pi_sc_14_23
                    
                #numu e^- nuebar <-> mu^-
                #nua b^- nubbar <-> a^-

                Ej = (j**2 + xb**2)**(1/2)
                El = i + Ej + k


                if (El > xa):

                    l = (El**2-xa**2)**(1/2)

                    fc_j = 1/(np.exp(Ej - zeta_b)+1) 
                    fc_j_bar = 1/(np.exp(Ej + zeta_b)+1) 
                    fc_l = 1/(np.exp(El - zeta_a)+1) 
                    fc_l_bar = 1/(np.exp(El + zeta_a)+1)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    Pi_sc_12_34 = D1 + D2_12/(i*Ej) - D2_34/(k*El) - D3/(i*Ej*k*El)

                    overall_fac = weight_simps*j*k*El

                    Coll += overall_fac  \
                                *(1-fc_l)*fc_j*f_nub_bar_k*4*Pi_sc_12_34   

                    Coll_bar += overall_fac  \
                                *(1-fc_l_bar)*fc_j_bar*f_nub_k*4*Pi_sc_12_34
                    
            else:

                #nutau e^- nuebar <-> tau^-
                #nua b^- nubbar <-> a^-

                Ej = (j**2 + xb**2)**(1/2)
                El = i + Ej + k


                if (El > xa):

                    l = (El**2-xa**2)**(1/2)

                    fc_j = 1/(np.exp(Ej - zeta_b)+1) 
                    fc_j_bar = 1/(np.exp(Ej + zeta_b)+1) 
                    fc_l = 1/(np.exp(El - zeta_a)+1) 
                    fc_l_bar = 1/(np.exp(El + zeta_a)+1)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    Pi_sc_12_34 = D1 + D2_12/(i*Ej) - D2_34/(k*El) - D3/(i*Ej*k*El)

                    overall_fac = weight_simps*j*k*El

                    Coll += overall_fac  \
                                *(1-fc_l)*fc_j*f_nub_bar_k*4*Pi_sc_12_34   

                    Coll_bar += overall_fac  \
                                *(1-fc_l_bar)*fc_j_bar*f_nub_k*4*Pi_sc_12_34
                    

                #nutau mu^- numubar <-> tau^-
                #nua c^- nucbar <-> a^-

                Ej = (j**2 + xc**2)**(1/2)
                El = i + Ej + k


                if (El > xa):

                    l = (El**2-xa**2)**(1/2)

                    fc_j = 1/(np.exp(Ej - zeta_c)+1) 
                    fc_j_bar = 1/(np.exp(Ej + zeta_c)+1) 
                    fc_l = 1/(np.exp(El - zeta_a)+1) 
                    fc_l_bar = 1/(np.exp(El + zeta_a)+1)

                    D1, D2_34, D2_12, D2_13, D2_14, D2_23, D2_24, D3 = D_function(i,j,k,l)

                    Pi_sc_12_34 = D1 + D2_12/(i*Ej) - D2_34/(k*El) - D3/(i*Ej*k*El)

                    overall_fac = weight_simps*j*k*El

                    Coll += overall_fac  \
                                *(1-fc_l)*fc_j*f_nuc_bar_k*4*Pi_sc_12_34   

                    Coll_bar += overall_fac  \
                                *(1-fc_l_bar)*fc_j_bar*f_nuc_k*4*Pi_sc_12_34
                    
    Coll *= T**5*GF**2/(2*np.pi**3*i)
    Coll_bar *= T**5*GF**2/(2*np.pi**3*i)
            

    return Coll, Coll_bar