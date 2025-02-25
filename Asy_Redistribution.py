import numpy as np
from numba import jit
from Constants import *
from Sterile_Nu_Parameters import *
from Momentum_Grid import *
from DOF import gs_below120MeV, gs_above120MeV
from Cor_Asy_QCD import Cor_Asy_QCD

ChiData = np.loadtxt("ChiTable.dat")

@jit(nopython=True,nogil=True,fastmath=True)
def eqs_redist(x,T,La):
    

    if (flavor == 'e'):

        Le = La
        Lmu = Lmu_ini
        Ltau = Ltau_ini

    elif (flavor == 'mu'):

        Le = Le_ini
        Lmu = La
        Ltau = Ltau_ini

    else:

        Le = Le_ini
        Lmu = Lmu_ini
        Ltau = La



    zeta_nue = x[0] #zeta = mu/T
    zeta_numu = x[1]
    zeta_nutau = x[2]
    zeta_B = x[3]
    zeta_Q = x[4]

    zeta_e = zeta_nue - zeta_Q
    zeta_mu = zeta_numu - zeta_Q   
    zeta_tau = zeta_nutau - zeta_Q


    zeta_u = 1/3*zeta_B + 2/3*zeta_Q
    zeta_d = 1/3*zeta_B - 1/3*zeta_Q
    zeta_c = 1/3*zeta_B + 2/3*zeta_Q
    zeta_s = 1/3*zeta_B - 1/3*zeta_Q

    zeta_pic = -zeta_Q

    zeta_p = 1/3*zeta_B + zeta_Q
    zeta_n = 1/3*zeta_B


    xe = me/T
    xmu = mmu/T
    xtau = mtau/T

    xu = mu/T
    xd = md/T
    xc = mc/T
    xs = ms/T

    xpic = mpic/T

    xp = mp/T
    xn = mn/T


    #the number density asymmetry

    Del_nue = 0
    Del_numu = 0
    Del_nutau = 0
    Del_e = 0
    Del_mu = 0
    Del_tau = 0

    Del_u = 0
    Del_d = 0
    Del_c = 0
    Del_s = 0

    Del_pic = 0

    Del_p = 0
    Del_n = 0

    #Corrections of chemical potnetial to entropy: s(mu) - s(mu=0)

    Del_s_nue = 0
    Del_s_numu = 0
    Del_s_nutau = 0
    Del_s_e = 0
    Del_s_mu = 0
    Del_s_tau = 0

    Del_s_u = 0
    Del_s_d = 0
    Del_s_c = 0
    Del_s_s = 0 

    Del_s_pic = 0



    #Effective DOF for entropy

    if (T<120):

        gs = gs_below120MeV(T)

    else:

        gs = gs_above120MeV(T)

    
    #Total entropy

    s_0 = gs*2*np.pi**2/45


    #We compute (n-n_bar)*s/T**3


    if (T >= 280):

        

        for ni in range(n3):

            y = y_min3 + dy3*ni
            weight = coe_simps3[ni]*y**2
            weight_s = coe_simps3[ni]

            Ee = (y**2 + xe**2)**(1/2)
            Emu = (y**2 + xmu**2)**(1/2)
            Etau = (y**2 + xtau**2)**(1/2)

            Eu = (y**2 + xu**2)**(1/2)
            Ed = (y**2 + xd**2)**(1/2)
            Ec = (y**2 + xc**2)**(1/2)
            Es = (y**2 + xs**2)**(1/2)

            Del_nue += weight*(1/(np.exp(y - zeta_nue) + 1) - 1/(np.exp(y + zeta_nue) + 1))
            Del_numu += weight*(1/(np.exp(y - zeta_numu) + 1) - 1/(np.exp(y + zeta_numu) + 1))
            Del_nutau += weight*(1/(np.exp(y - zeta_nutau) + 1) - 1/(np.exp(y + zeta_nutau) + 1))

            Del_e += weight*(1/(np.exp(Ee-zeta_e) + 1) - 1/(np.exp(Ee+zeta_e) + 1))
            Del_mu += weight*(1/(np.exp(Emu - zeta_mu) + 1) - 1/(np.exp(Emu + zeta_mu) + 1))
            Del_tau += weight*(1/(np.exp(Etau - zeta_tau) + 1) - 1/(np.exp(Etau + zeta_tau) + 1))

            Del_u += weight*(1/(np.exp(Eu - zeta_u) + 1) - 1/(np.exp(Eu + zeta_u) + 1))
            Del_d += weight*(1/(np.exp(Ed - zeta_d) + 1) - 1/(np.exp(Ed + zeta_d) + 1))
            Del_c += weight*(1/(np.exp(Ec- zeta_c) + 1) - 1/(np.exp(Ec + zeta_c) + 1))
            Del_s += weight*(1/(np.exp(Es - zeta_s) + 1) - 1/(np.exp(Es + zeta_s) + 1))

            Del_s_nue += weight_s*((4/3*y**3 -zeta_nue*y**2)*(1/(np.exp(y - zeta_nue) + 1)) + (4/3*y**3 + zeta_nue*y**2)*(1/(np.exp(y + zeta_nue) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            Del_s_numu += weight_s*((4/3*y**3 -zeta_numu*y**2)*(1/(np.exp(y - zeta_numu) + 1)) + (4/3*y**3 + zeta_numu*y**2)*(1/(np.exp(y + zeta_numu) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            Del_s_nutau += weight_s*((4/3*y**3 -zeta_nutau*y**2)*(1/(np.exp(y - zeta_nutau) + 1)) + (4/3*y**3 + zeta_nutau*y**2)*(1/(np.exp(y + zeta_nutau) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            
            Del_s_e += weight_s*((y**2*Ee + y**4/(3*Ee) -zeta_e*y**2)*(1/(np.exp(Ee - zeta_e) + 1)) + (y**2*Ee + y**4/(3*Ee) + zeta_e*y**2)*(1/(np.exp(Ee + zeta_e) + 1)) \
                                      - 2*(y**2*Ee + y**4/(3*Ee))*(1/(np.exp(Ee) + 1)))
            Del_s_mu += weight_s*((y**2*Emu + y**4/(3*Emu) -zeta_mu*y**2)*(1/(np.exp(Emu - zeta_mu) + 1)) + (y**2*Emu + y**4/(3*Emu) + zeta_mu*y**2)*(1/(np.exp(Emu + zeta_mu) + 1)) \
                                      - 2*(y**2*Emu + y**4/(3*Emu))*(1/(np.exp(Emu) + 1)))
            Del_s_tau += weight_s*((y**2*Etau + y**4/(3*Etau) -zeta_tau*y**2)*(1/(np.exp(Etau - zeta_tau) + 1)) + (y**2*Etau + y**4/(3*Etau) + zeta_tau*y**2)*(1/(np.exp(Etau + zeta_tau) + 1)) \
                                      - 2*(y**2*Etau + y**4/(3*Etau))*(1/(np.exp(Etau) + 1)))
            
            Del_s_u += weight_s*((y**2*Eu + y**4/(3*Eu) -zeta_u*y**2)*(1/(np.exp(Eu - zeta_u) + 1)) + (y**2*Eu + y**4/(3*Eu) + zeta_u*y**2)*(1/(np.exp(Eu + zeta_u) + 1)) \
                                      - 2*(y**2*Eu + y**4/(3*Eu))*(1/(np.exp(Eu) + 1)))
            Del_s_d += weight_s*((y**2*Ed + y**4/(3*Ed) -zeta_d*y**2)*(1/(np.exp(Ed - zeta_d) + 1)) + (y**2*Ed + y**4/(3*Ed) + zeta_d*y**2)*(1/(np.exp(Ed + zeta_d) + 1)) \
                                      - 2*(y**2*Ed + y**4/(3*Ed))*(1/(np.exp(Ed) + 1)))
            Del_s_c += weight_s*((y**2*Ec + y**4/(3*Ec) -zeta_c*y**2)*(1/(np.exp(Ec - zeta_c) + 1)) + (y**2*Ec + y**4/(3*Ec) + zeta_c*y**2)*(1/(np.exp(Ec + zeta_c) + 1)) \
                                      - 2*(y**2*Ec + y**4/(3*Ec))*(1/(np.exp(Ec) + 1)))
            Del_s_s += weight_s*((y**2*Es + y**4/(3*Es) -zeta_s*y**2)*(1/(np.exp(Es - zeta_s) + 1)) + (y**2*Es + y**4/(3*Es) + zeta_s*y**2)*(1/(np.exp(Es + zeta_s) + 1)) \
                                      - 2*(y**2*Es + y**4/(3*Es))*(1/(np.exp(Es) + 1)))


        Del_nue *= 1/(2*np.pi**2)
        Del_numu *= 1/(2*np.pi**2)
        Del_nutau *= 1/(2*np.pi**2)

        Del_e *= 2/(2*np.pi**2)
        Del_mu *= 2/(2*np.pi**2)
        Del_tau *= 2/(2*np.pi**2)


        Del_u *= 6/(2*np.pi**2)
        Del_d *= 6/(2*np.pi**2)
        Del_c *= 6/(2*np.pi**2)
        Del_s *= 6/(2*np.pi**2)

        s = s_0 + 1/(2*np.pi**2)*(Del_s_nue + Del_s_numu + Del_s_nutau + 2*(Del_s_e + Del_s_mu + Del_s_tau) + 6*(Del_s_u + Del_s_d + Del_s_c + Del_s_s))
        #s = s_0

        Cor_Del_B, Cor_Del_C = Cor_Asy_QCD(T,zeta_u,zeta_c,zeta_d,zeta_s)
        
        #Cor_Del_B = 0
        #Cor_Del_C =0



        f = np.zeros(5)
        f[0] = (Del_nue + Del_e) - Le*s  #e asymmetry
        f[1] = (Del_numu + Del_mu) - Lmu*s #mu asymmetry
        f[2] = (Del_nutau + Del_tau) - Ltau*s  #tau asymmetry
        #f[3] = 1/3*(Del_u + Del_d + Del_c + Del_s) + Cor_Del_B - 8.6e-11*s #baryon asymmetry
        f[3] = 1/3*(Del_u + Del_d + Del_c + Del_s) + Cor_Del_B #baryon asymmetry
        f[4] = -1*(Del_e + Del_mu + Del_tau) + 2/3*(Del_u + Del_c) - 1/3*(Del_d + Del_s) + Cor_Del_C #Electric charge asymmetry

    elif (T < 280) and (T >= 120):

        #Fixing values of susceptibilities

        TIdx = np.searchsorted(ChiData[:, 0], T)

        Chi2BSlope = (ChiData[TIdx, 1] - ChiData[TIdx - 1, 1]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi2B = Chi2BSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 1]

        Chi2QSlope = (ChiData[TIdx, 2] - ChiData[TIdx - 1, 2]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi2Q = Chi2QSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 2]

        Chi11QBSlope = (ChiData[TIdx, 3] - ChiData[TIdx - 1, 3]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi11QB = Chi11QBSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 3]

        for ni in range(n3):

            y = y_min3 + dy3*ni
            weight = coe_simps3[ni]*y**2
            weight_s = coe_simps3[ni]

            Ee = (y**2 + xe**2)**(1/2)
            Emu = (y**2 + xmu**2)**(1/2)
            Etau = (y**2 + xtau**2)**(1/2)

            Del_nue += weight*(1/(np.exp(y - zeta_nue) + 1) - 1/(np.exp(y + zeta_nue) + 1))
            Del_numu += weight*(1/(np.exp(y - zeta_numu) + 1) - 1/(np.exp(y + zeta_numu) + 1))
            Del_nutau += weight*(1/(np.exp(y - zeta_nutau) + 1) - 1/(np.exp(y + zeta_nutau) + 1))

            Del_e += weight*(1/(np.exp(Ee-zeta_e) + 1) - 1/(np.exp(Ee+zeta_e) + 1))
            Del_mu += weight*(1/(np.exp(Emu - zeta_mu) + 1) - 1/(np.exp(Emu + zeta_mu) + 1))
            Del_tau += weight*(1/(np.exp(Etau - zeta_tau) + 1) - 1/(np.exp(Etau + zeta_tau) + 1))

            Del_s_nue += weight_s*((4/3*y**3 -zeta_nue*y**2)*(1/(np.exp(y - zeta_nue) + 1)) + (4/3*y**3 + zeta_nue*y**2)*(1/(np.exp(y + zeta_nue) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            Del_s_numu += weight_s*((4/3*y**3 -zeta_numu*y**2)*(1/(np.exp(y - zeta_numu) + 1)) + (4/3*y**3 + zeta_numu*y**2)*(1/(np.exp(y + zeta_numu) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            Del_s_nutau += weight_s*((4/3*y**3 -zeta_nutau*y**2)*(1/(np.exp(y - zeta_nutau) + 1)) + (4/3*y**3 + zeta_nutau*y**2)*(1/(np.exp(y + zeta_nutau) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            
            Del_s_e += weight_s*((y**2*Ee + y**4/(3*Ee) -zeta_e*y**2)*(1/(np.exp(Ee - zeta_e) + 1)) + (y**2*Ee + y**4/(3*Ee) + zeta_e*y**2)*(1/(np.exp(Ee + zeta_e) + 1)) \
                                      - 2*(y**2*Ee + y**4/(3*Ee))*(1/(np.exp(Ee) + 1)))
            Del_s_mu += weight_s*((y**2*Emu + y**4/(3*Emu) -zeta_mu*y**2)*(1/(np.exp(Emu - zeta_mu) + 1)) + (y**2*Emu + y**4/(3*Emu) + zeta_mu*y**2)*(1/(np.exp(Emu + zeta_mu) + 1)) \
                                      - 2*(y**2*Emu + y**4/(3*Emu))*(1/(np.exp(Emu) + 1)))
            Del_s_tau += weight_s*((y**2*Etau + y**4/(3*Etau) -zeta_tau*y**2)*(1/(np.exp(Etau - zeta_tau) + 1)) + (y**2*Etau + y**4/(3*Etau) + zeta_tau*y**2)*(1/(np.exp(Etau + zeta_tau) + 1)) \
                                      - 2*(y**2*Etau + y**4/(3*Etau))*(1/(np.exp(Etau) + 1)))
            
        Del_nue *= 1/(2*np.pi**2)
        Del_numu *= 1/(2*np.pi**2)
        Del_nutau *= 1/(2*np.pi**2)

        Del_e *= 2/(2*np.pi**2)
        Del_mu *= 2/(2*np.pi**2)
        Del_tau *= 2/(2*np.pi**2)
        

        #Corrections to the total enetropy

        s = s_0 + 1/2*T*(Chi2BSlope*zeta_B**2 + 2*Chi11QBSlope*zeta_B*zeta_Q + Chi2QSlope*zeta_Q**2) + 1/(2*np.pi**2)*(Del_s_nue + Del_s_numu + Del_s_nutau + 2*(Del_s_e + Del_s_mu + Del_s_tau))
        #s = s_0

        f = np.zeros(5)
        f[0] = (Del_nue + Del_e) - Le*s #e asymmetry
        f[1] = (Del_numu + Del_mu) - Lmu*s 
        f[2] = (Del_nutau + Del_tau) - Ltau*s #tau asymmetry
        #f[3] = Chi11QB*zeta_Q + Chi2B*zeta_B - 8.6e-11*s
        f[3] = Chi11QB*zeta_Q + Chi2B*zeta_B
        f[4] = Chi2Q*zeta_Q + Chi11QB*zeta_B - (Del_e + Del_mu + Del_tau)

    else:

        for ni in range(n3):

            y = y_min3 + dy3*ni
            weight = coe_simps3[ni]*y**2
            weight_s = coe_simps3[ni]

            Ee = (y**2 + xe**2)**(1/2)
            Emu = (y**2 + xmu**2)**(1/2)
            Etau = (y**2 + xtau**2)**(1/2)

            Epic = (y**2 + xpic**2)**(1/2)

            Del_nue += weight*(1/(np.exp(y - zeta_nue) + 1) - 1/(np.exp(y + zeta_nue) + 1))
            Del_numu += weight*(1/(np.exp(y - zeta_numu) + 1) - 1/(np.exp(y + zeta_numu) + 1))
            Del_nutau += weight*(1/(np.exp(y - zeta_nutau) + 1) - 1/(np.exp(y + zeta_nutau) + 1))

            Del_e += weight*(1/(np.exp(Ee - zeta_e) + 1) - 1/(np.exp(Ee + zeta_e) + 1))
            Del_mu += weight*(1/(np.exp(Emu - zeta_mu) + 1) - 1/(np.exp(Emu + zeta_mu) + 1))
            Del_tau += weight*(1/(np.exp(Etau - zeta_tau) + 1) - 1/(np.exp(Etau + zeta_tau) + 1))

            Del_pic += weight*(1/(np.exp(Epic - zeta_pic) - 1) - 1/(np.exp(Epic + zeta_pic) - 1))

            Del_s_nue += weight_s*((4/3*y**3 -zeta_nue*y**2)*(1/(np.exp(y - zeta_nue) + 1)) + (4/3*y**3 + zeta_nue*y**2)*(1/(np.exp(y + zeta_nue) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            Del_s_numu += weight_s*((4/3*y**3 -zeta_numu*y**2)*(1/(np.exp(y - zeta_numu) + 1)) + (4/3*y**3 + zeta_numu*y**2)*(1/(np.exp(y + zeta_numu) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            Del_s_nutau += weight_s*((4/3*y**3 -zeta_nutau*y**2)*(1/(np.exp(y - zeta_nutau) + 1)) + (4/3*y**3 + zeta_nutau*y**2)*(1/(np.exp(y + zeta_nutau) + 1)) \
                                      - 2*(4/3*y**3)*(1/(np.exp(y) + 1)))
            
            Del_s_e += weight_s*((y**2*Ee + y**4/(3*Ee) -zeta_e*y**2)*(1/(np.exp(Ee - zeta_e) + 1)) + (y**2*Ee + y**4/(3*Ee) + zeta_e*y**2)*(1/(np.exp(Ee + zeta_e) + 1)) \
                                      - 2*(y**2*Ee + y**4/(3*Ee))*(1/(np.exp(Ee) + 1)))
            Del_s_mu += weight_s*((y**2*Emu + y**4/(3*Emu) -zeta_mu*y**2)*(1/(np.exp(Emu - zeta_mu) + 1)) + (y**2*Emu + y**4/(3*Emu) + zeta_mu*y**2)*(1/(np.exp(Emu + zeta_mu) + 1)) \
                                      - 2*(y**2*Emu + y**4/(3*Emu))*(1/(np.exp(Emu) + 1)))
            Del_s_tau += weight_s*((y**2*Etau + y**4/(3*Etau) -zeta_tau*y**2)*(1/(np.exp(Etau - zeta_tau) + 1)) + (y**2*Etau + y**4/(3*Etau) + zeta_tau*y**2)*(1/(np.exp(Etau + zeta_tau) + 1)) \
                                      - 2*(y**2*Etau + y**4/(3*Etau))*(1/(np.exp(Etau) + 1)))
            
            Del_s_pic += weight_s*((y**2*Epic + y**4/(3*Epic) -zeta_pic*y**2)*(1/(np.exp(Epic - zeta_pic) - 1)) + (y**2*Epic + y**4/(3*Epic) + zeta_pic*y**2)*(1/(np.exp(Epic + zeta_pic) - 1)) \
                                      - 2*(y**2*Epic + y**4/(3*Epic))*(1/(np.exp(Epic) - 1)))

            


        Del_p = 2*(mp*T/(2*np.pi))**(3/2)*np.exp(-xp)*2*zeta_p/T**3
        Del_n = 2*(mn*T/(2*np.pi))**(3/2)*np.exp(-xn)*2*zeta_n/T**3


        Del_nue *= 1/(2*np.pi**2)
        Del_numu *= 1/(2*np.pi**2)
        Del_nutau *= 1/(2*np.pi**2)

        Del_e *= 2/(2*np.pi**2)
        Del_mu *= 2/(2*np.pi**2)
        Del_tau *= 2/(2*np.pi**2)

        Del_pic *= 1/(2*np.pi**2)

        s = s_0 + 1/(2*np.pi**2)*(Del_s_nue + Del_s_numu + Del_s_nutau + 2*(Del_s_e + Del_s_mu + Del_s_tau) + Del_s_pic)
        #s = s_0



        f = np.zeros(5)
        f[0] = (Del_nue + Del_e) - Le*s #e asymmetry
        f[1] = (Del_numu + Del_mu) - Lmu*s 
        f[2] = (Del_nutau + Del_tau) - Ltau*s #tau asymmetry
        #f[3] = Del_p + Del_n - 8.6e-11*s #baryon asymmetry
        f[3] = Del_p + Del_n #baryon asymmetry
        f[4] = -(Del_e + Del_mu + Del_tau + Del_pic) + Del_p #Electric charge asymmetry
        #f[4] = -(Del_e + Del_mu + Del_tau + Del_pic) #Electric charge asymmetry
            

    
    return f



