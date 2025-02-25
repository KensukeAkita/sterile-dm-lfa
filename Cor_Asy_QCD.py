import numpy as np
from numba import jit
from Collision_term.D_function import D_function
from Constants import *
from Sterile_Nu_Parameters import *
from Momentum_Grid import *


@jit(nopython=True,nogil=True,fastmath=True) 
def Cor_Asy_QCD(T,zeta_u,zeta_c,zeta_d,zeta_s):

    #Finite temperature QCD corrections

    #Corrrection to coupling and quark masses by 1-loop running 

    mubar = 4*np.pi*T*np.exp(-GammaE+(-Nc+4*Nf*np.log(4))/(22*Nc-4*Nf))

    gstrong2 = 24*np.pi**2/((11*CA-4*TF)*np.log(mubar/LambdaMS)) #strong coupling squared

    m_cor = (np.log(muref/LambdaMS)/np.log(mubar/LambdaMS))**(9*CF/(11*CA-4*TF))


    mu_cor = mu*m_cor
    md_cor = md*m_cor
    mc_cor = mc*m_cor #Subtle
    ms_cor = ms*m_cor

    #Correction to asymmetry of number density

    Cor_Del_B = 0
    Cor_Del_C = 0

    xQ_list = [mu_cor/T, mc_cor/T, md_cor/T, ms_cor/T]
    zetaQ_list = [zeta_u, zeta_c, zeta_d, zeta_s]
    fac_B = 1/3
    fac_C_list = [2/3, 2/3, -1/3, -1/3]


    alphaE2 = 0
    dalphaE2dmu = 0


    alphaE7 = 0
    dalphaE7dmu = 0

    for nQ in range(4):

        xQ = xQ_list[nQ] 
        zetaQ = zetaQ_list[nQ]

        fac_C = fac_C_list[nQ]


        F2 = 0
        F3 = 0
        F4 = 0

        dF2dmu = 0
        dF4dmu = 0
        dF3dmu = 0
            

        for ni in range(n4):

            y = y_min4 + dy4*ni
            y1 = y

            fQ = 1/(np.exp((y+xQ**2)**(1/2)-zetaQ)+1)
            fQ_bar = 1/(np.exp((y+xQ**2)**(1/2)+zetaQ)+1)
            fQ1 = fQ
            fQ_bar1 = fQ_bar 

            dfQdmu = np.exp((y+xQ**2)**(1/2)-zetaQ)/(np.exp((y+xQ**2)**(1/2)-zetaQ)+1)**2
            dfQ_bardmu = -np.exp((y+xQ**2)**(1/2)+zetaQ)/(np.exp((y+xQ**2)**(1/2)+zetaQ)+1)**2

            dfdmu1 = dfQdmu
            df_bardmu1 = dfQ_bardmu

            F2 += coe_simps4[ni]*1/(8*np.pi**2)*y*(y/(y+xQ**2))**(1/2)*(fQ + fQ_bar)

            F3 += -coe_simps4[ni]*1/y*(y/(y+xQ**2))**(1/2)*(fQ + fQ_bar)

            dF2dmu += coe_simps4[ni]*1/(8*np.pi**2)*y*(y/(y+xQ**2))**(1/2)*(dfQdmu + dfQ_bardmu)

            dF3dmu + -coe_simps4[ni]*1/y*(y/(y+xQ**2))**(1/2)*(dfQdmu + dfQ_bardmu)

            for nj in range(n4):

                y2 = y_min4 + dy4*nj

                fQ2 = 1/(np.exp((y2+xQ**2)**(1/2)-zetaQ)+1)
                fQ_bar2 = 1/(np.exp((y2+xQ**2)**(1/2)+zetaQ)+1)

                dfdmu2 = np.exp((y2+xQ**2)**(1/2)-zetaQ)/(np.exp((y2+xQ**2)**(1/2)-zetaQ)+1)**2
                df_bardmu2 = -np.exp((y2+xQ**2)**(1/2)+zetaQ)/(np.exp((y2+xQ**2)**(1/2)+zetaQ)+1)**2

                F4 += coe_simps4[ni]*coe_simps4[nj]*1/(4*np.pi)**4*1/((y1+xQ**2)*(y2+xQ**2))**(1/2) \
                        *((fQ1*fQ_bar2 + fQ1*fQ_bar2 + fQ_bar1*fQ2 + fQ_bar1*fQ2) \
                          *np.log((((y1+xQ**2)*(y2+xQ**2))**(1/2) + xQ**2 - (y1*y2)**(1/2)) / (((y1+xQ**2)*(y2+xQ**2))**(1/2) + xQ**2 + (y1*y2)**(1/2))) \
                        + (fQ1*fQ2 + fQ1*fQ2 + fQ_bar1*fQ_bar2 + fQ_bar1*fQ_bar2) \
                            *np.log((((y1+xQ**2)*(y2+xQ**2))**(1/2) - xQ**2 + (y1*y2)**(1/2)) / (((y1+xQ**2)*(y2+xQ**2))**(1/2) - xQ**2 - (y1*y2)**(1/2) + 1e-12))) #Cutoff factor is necessary
                
                dF4dmu += coe_simps4[ni]*coe_simps4[nj]*1/(4*np.pi)**4*1/((y1+xQ**2)*(y2+xQ**2))**(1/2) \
                        *((dfdmu1*fQ_bar2 + fQ1*df_bardmu2 + df_bardmu1*fQ2 + fQ_bar1*dfdmu2) \
                          *np.log((((y1+xQ**2)*(y2+xQ**2))**(1/2) + xQ**2 - (y1*y2)**(1/2)) / (((y1+xQ**2)*(y2+xQ**2))**(1/2) + xQ**2 + (y1*y2)**(1/2))) \
                        + (dfdmu1*fQ2 + fQ1*dfdmu2 + df_bardmu1*fQ_bar2 + fQ_bar1*df_bardmu2) \
                            *np.log((((y1+xQ**2)*(y2+xQ**2))**(1/2) - xQ**2 + (y1*y2)**(1/2)) / (((y1+xQ**2)*(y2+xQ**2))**(1/2) - xQ**2 - (y1*y2)**(1/2) + 1e-12))) #Cutoff factor is necessary
                
        
        alphaE2 += -dA*(1/6*F2*(1+6*F2) + xQ**2/(4*np.pi**2)*(3*np.log(mubar/(xQ*T)) + 2)*F2 -2*xQ**2*F4) 
        dalphaE2dmu += -dA*((1/6*dF2dmu*(1+6*F2) + F2*dF2dmu) + xQ**2/(4*np.pi**2)*(3*np.log(mubar/(xQ*T)) + 2)*dF2dmu -2*xQ**2*dF4dmu) 
        alphaE7 += 2*np.log(mubar/xQ*T) + F3
        dalphaE7dmu += -2/3*dF3dmu

    
    alphaE2 = alphaE2 - dA*CA/144
    alphaE7 = 22*CA/3*(np.log(mubar*np.exp(GammaE)/(4*np.pi*T)) + 1/22) - 2/3*alphaE7


    g32 = gstrong2 + gstrong2**2/(4*np.pi)**2*alphaE7

    dg32dmu = gstrong2**2/(4*np.pi)**2*dalphaE7dmu

    

    Cor_Del_B = fac_B*(g32*dalphaE2dmu + dg32dmu*alphaE2)
    Cor_Del_C = fac_C*(g32*dalphaE2dmu + dg32dmu*alphaE2)

    return Cor_Del_B, Cor_Del_C