"""
Thermodynamics quantities (energy density, entropy density, pressure, number density asymmetries) are calculated. 
All Thermodynamics quantities are normalized by temperature.
"""
import numpy as np
from numba import jit
from Constants import *
from QCD_GlobalParameters import * 
from Setup_Grids import *
from Thermodynamics.DOF import *

ChiData = np.loadtxt("ChiTable.dat")

@jit(nopython=True,nogil=True,fastmath=True)
def Thermo_quantities(T,zeta_nue,zeta_numu,zeta_nutau,zeta_B,zeta_Q,flavor):

    zeta_e = zeta_nue - zeta_Q #chemical potential normalized by temperature, zeta = mu/T
    zeta_mu = zeta_numu - zeta_Q   
    zeta_tau = zeta_nutau - zeta_Q

    zeta_u = 1/3*zeta_B + 2/3*zeta_Q
    zeta_d = 1/3*zeta_B - 1/3*zeta_Q
    zeta_c = 1/3*zeta_B + 2/3*zeta_Q
    zeta_s = 1/3*zeta_B - 1/3*zeta_Q

    zeta_pic = -zeta_Q


    xe = me/T
    xmu = mmu/T
    xtau = mtau/T

    xu = mu/T
    xd = md/T
    xc = mc/T
    xs = ms/T

    xpic = mpic/T

    if (flavor == 'e'):

        zeta_nua = zeta_nue
        zeta_a = zeta_e

        ma = me

    elif (flavor == 'mu'):

        zeta_nua = zeta_numu
        zeta_a = zeta_mu

        ma = mmu

    else:

        zeta_nua = zeta_nutau
        zeta_a = zeta_tau

        ma = mtau

    ################################################################
    #Total energy density and entropy at zero chemical potential
    ################################################################

    #Effective DOF

    if (T<T_hadron):

        gs = gs_below120MeV(T)
        grho = grho_below120MeV(T)

    else:

        gs = gs_above120MeV(T)
        grho = grho_above120MeV(T)


    s_0 = gs*2*np.pi**2/45

    rho_0 = grho*np.pi**2/30


    ################################################################
    #Computing energy density and lepton asymmetry in lepton sector  
    ################################################################

    #Energy density and number density

    rho_nua = 0
    rho_a = 0

    #Lepton (flavor) asymmetry in terms of number density: n - n_bar

    Del_nua = 0 #nua asymmetry
    Del_a = 0

    Del_nu_total = 0
    Del_c_total = 0

    for ni in range(n2):

        y = y_min2 + dy2*ni
        weight = coe_simps2[ni]*y**2

        Ea = (y**2 + (ma/T)**2)**(1/2)
        Ee = (y**2 + xe**2)**(1/2)
        Emu = (y**2 + xmu**2)**(1/2)
        Etau = (y**2 + xtau**2)**(1/2)

        rho_nua += weight*y*(1/(np.exp(y - zeta_nua) + 1) + 1/(np.exp(y + zeta_nua) + 1))
        rho_a += weight*Ea*(1/(np.exp(Ea - zeta_a) + 1) +  1/(np.exp(Ea + zeta_a) + 1))

        Del_nua += weight*(1/(np.exp(y - zeta_nua) + 1) - 1/(np.exp(y + zeta_nua) + 1))
        Del_nu_total += weight*(1/(np.exp(y - zeta_nue) + 1) + 1/(np.exp(y - zeta_numu) + 1) + 1/(np.exp(y - zeta_nutau) + 1) \
                                            - 1/(np.exp(y + zeta_nue) + 1) - 1/(np.exp(y + zeta_numu) + 1) - 1/(np.exp(y + zeta_nutau) + 1))
        Del_a += weight*(1/(np.exp(Ea - zeta_a) + 1) -  1/(np.exp(Ea + zeta_a) + 1))
        Del_c_total += weight*(1/(np.exp(Ee - zeta_e) + 1) -  1/(np.exp(Ee + zeta_e) + 1) \
                                                + 1/(np.exp(Emu - zeta_mu) + 1) -  1/(np.exp(Emu + zeta_mu) + 1) \
                                                + 1/(np.exp(Etau - zeta_tau) + 1) -  1/(np.exp(Etau + zeta_tau) + 1))   


    rho_nua *= 1/(2*np.pi**2) #nu and anti-nu corresponding flavor mixing with sterile nu
    rho_a *= 2/(2*np.pi**2) #charged leptons and their anti-leptons corresponding flavor mixing with sterile nu

    Del_nua *= 1/(2*np.pi**2)

    Del_nu_total *= 1/(2*np.pi**2)

    Del_a *= 2/(2*np.pi**2) #2 is d.o.f for spin

    Del_c_total *= 2/(2*np.pi**2) #2 is d.o.f for spin



    ###########################################################
    #Computing electric charge asymmetry in quark/hadron sector
    ###########################################################

    #Electric charge asymmetry for quark/hadron sector in terms of number density: Q*(n - n_bar)

    Del_Q_QCD = 0

    if (T >= T_quark):

        Del_u = 0
        Del_d = 0
        Del_c = 0
        Del_s = 0 

        for ni in range(n2):

            y = y_min2 + dy2*ni
            weight = coe_simps2[ni]*y**2

            Eu = (y**2 + xu**2)**(1/2)
            Ed = (y**2 + xd**2)**(1/2)
            Ec = (y**2 + xc**2)**(1/2)
            Es = (y**2 + xs**2)**(1/2)

            Del_u += weight*(1/(np.exp(Eu - zeta_u) + 1) - 1/(np.exp(Eu + zeta_u) + 1))
            Del_d += weight*(1/(np.exp(Ed - zeta_d) + 1) - 1/(np.exp(Ed + zeta_d) + 1))
            Del_c += weight*(1/(np.exp(Ec - zeta_c) + 1) - 1/(np.exp(Ec + zeta_c) + 1))
            Del_s += weight*(1/(np.exp(Es - zeta_s) + 1) - 1/(np.exp(Es + zeta_s) + 1))
        
        Del_Q_QCD = 6/(2*np.pi**2)*(2/3*Del_u - 1/3*Del_d + 2/3*Del_c - 1/3*Del_s)



    elif (T < T_quark) and (T >= T_hadron):

        TIdx = np.searchsorted(ChiData[:, 0], T)

        Chi2QSlope = (ChiData[TIdx, 2] - ChiData[TIdx - 1, 2]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi2Q = Chi2QSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 2]

        Chi11QBSlope = (ChiData[TIdx, 3] - ChiData[TIdx - 1, 3]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi11QB = Chi11QBSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 3]

        Del_Q_QCD = (Chi2Q*zeta_Q + Chi11QB*zeta_B)


    else:

        Del_pic = 0

        for ni in range(n2):

            y = y_min2 + dy2*ni
            weight = coe_simps2[ni]*y**2

            Epic = (y**2 + xpic**2)**(1/2)

            Del_pic += weight*(1/(np.exp(Epic - zeta_pic) - 1) - 1/(np.exp(Epic + zeta_pic) - 1))

        Del_Q_QCD = 1/(2*np.pi**2)*(-Del_pic)



    #################################################################
    #Computing corrections to thermo quantities by large chemical potential
    #################################################################


    #Corrections of chemical potential to energy density (particl + anti-particle)

    Del_rho_nue = 0
    Del_rho_numu = 0
    Del_rho_nutau = 0
    Del_rho_e = 0
    Del_rho_mu = 0
    Del_rho_tau = 0

    Del_rho_u = 0
    Del_rho_d = 0
    Del_rho_c = 0
    Del_rho_s = 0 

    Del_rho_pic = 0

    #Corrections of chemical potential to Pressure (particl + anti-particle)

    N_nue = 0
    N_numu = 0
    N_nutau = 0
    N_e = 0
    N_mu = 0
    N_tau = 0

    N_u = 0
    N_d = 0
    N_c = 0
    N_s = 0 

    N_pic = 0

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


    if (T >= T_quark):

        

        for ni in range(n2):

            y = y_min2 + dy2*ni
            weight = coe_simps2[ni]*y**2
            weight_s = coe_simps2[ni]

            Ee = (y**2 + xe**2)**(1/2)
            Emu = (y**2 + xmu**2)**(1/2)
            Etau = (y**2 + xtau**2)**(1/2)

            Eu = (y**2 + xu**2)**(1/2)
            Ed = (y**2 + xd**2)**(1/2)
            Ec = (y**2 + xc**2)**(1/2)
            Es = (y**2 + xs**2)**(1/2)

            Del_rho_nue += weight*y*(1/(np.exp(y - zeta_nue) + 1) + 1/(np.exp(y + zeta_nue) + 1) - 2*1/(np.exp(y) + 1))
            Del_rho_numu += weight*y*(1/(np.exp(y - zeta_numu) + 1) + 1/(np.exp(y + zeta_numu) + 1) - 2*1/(np.exp(y) + 1))
            Del_rho_nutau += weight*y*(1/(np.exp(y - zeta_nutau) + 1) + 1/(np.exp(y + zeta_nutau) + 1) - 2*1/(np.exp(y) + 1))

            Del_rho_e += weight*Ee*(1/(np.exp(Ee - zeta_e) + 1) + 1/(np.exp(Ee + zeta_e) + 1) - 2*1/(np.exp(Ee) + 1))
            Del_rho_mu += weight*Emu*(1/(np.exp(Emu - zeta_mu) + 1) + 1/(np.exp(Emu + zeta_mu) + 1) - 2*1/(np.exp(Emu) + 1))
            Del_rho_tau += weight*Etau*(1/(np.exp(Etau - zeta_tau) + 1) + 1/(np.exp(Etau + zeta_tau) + 1) - 2*1/(np.exp(Etau) + 1))

            Del_rho_u += weight*Eu*(1/(np.exp(Eu - zeta_u) + 1) + 1/(np.exp(Eu + zeta_u) + 1) - 2*1/(np.exp(Eu) + 1))
            Del_rho_d += weight*Ed*(1/(np.exp(Ed - zeta_d) + 1) + 1/(np.exp(Ed + zeta_d) + 1) - 2*1/(np.exp(Ed) + 1))
            Del_rho_c += weight*Ec*(1/(np.exp(Ec - zeta_c) + 1) + 1/(np.exp(Ec + zeta_c) + 1) - 2*1/(np.exp(Ec) + 1))
            Del_rho_s += weight*Es*(1/(np.exp(Es - zeta_s) + 1) + 1/(np.exp(Es + zeta_s) + 1) - 2*1/(np.exp(Es) + 1))

            N_nue += weight*(1/(np.exp(y - zeta_nue) + 1) - 1/(np.exp(y + zeta_nue) + 1))
            N_numu += weight*(1/(np.exp(y - zeta_numu) + 1) - 1/(np.exp(y + zeta_numu) + 1))
            N_nutau += weight*(1/(np.exp(y - zeta_nutau) + 1) - 1/(np.exp(y + zeta_nutau) + 1))

            N_e += weight*(1/(np.exp(Ee - zeta_e) + 1) - 1/(np.exp(Ee + zeta_e) + 1))
            N_mu += weight*(1/(np.exp(Emu - zeta_mu) + 1) - 1/(np.exp(Emu + zeta_mu) + 1))
            N_tau += weight*(1/(np.exp(Etau - zeta_tau) + 1) - 1/(np.exp(Etau + zeta_tau) + 1))

            N_u += weight*(1/(np.exp(Eu - zeta_u) + 1) - 1/(np.exp(Eu + zeta_u) + 1))
            N_d += weight*(1/(np.exp(Ed - zeta_d) + 1) - 1/(np.exp(Ed + zeta_d) + 1))
            N_c += weight*(1/(np.exp(Ec - zeta_c) + 1) - 1/(np.exp(Ec + zeta_c) + 1))
            N_s += weight*(1/(np.exp(Es - zeta_s) + 1) - 1/(np.exp(Es + zeta_s) + 1))


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


        s = s_0 + 1/(2*np.pi**2)*(Del_s_nue + Del_s_numu + Del_s_nutau + 2*(Del_s_e + Del_s_mu + Del_s_tau) + 6*(Del_s_u + Del_s_d + Del_s_c + Del_s_s))
        Delta_rho = 1/(2*np.pi**2)*(Del_rho_nue + Del_rho_numu + Del_rho_nutau + 2*(Del_rho_e + Del_rho_mu + Del_rho_tau) + 6*(Del_rho_u + Del_rho_d + Del_rho_c + Del_rho_s))
        rho = rho_0 + Delta_rho
        
        P = s - rho +  1/(2*np.pi**2)*(zeta_nue*N_nue + zeta_numu*N_numu + zeta_nutau*N_nutau + 2*(zeta_e*N_e + zeta_mu*N_mu + zeta_tau*N_tau) + 6*(zeta_u*N_u + zeta_d*N_d + zeta_c*N_c + zeta_s*N_s))

        #s = s_0
        #rho = rho_0
        #P = s - rho


    elif (T < T_quark) and (T >= T_hadron):

        for ni in range(n2):

            y = y_min2 + dy2*ni
            weight = coe_simps2[ni]*y**2
            weight_s = coe_simps2[ni]

            Ee = (y**2 + xe**2)**(1/2)
            Emu = (y**2 + xmu**2)**(1/2)
            Etau = (y**2 + xtau**2)**(1/2)


            Del_rho_nue += weight*y*(1/(np.exp(y - zeta_nue) + 1) + 1/(np.exp(y + zeta_nue) + 1) - 2*1/(np.exp(y) + 1))
            Del_rho_numu += weight*y*(1/(np.exp(y - zeta_numu) + 1) + 1/(np.exp(y + zeta_numu) + 1) - 2*1/(np.exp(y) + 1))
            Del_rho_nutau += weight*y*(1/(np.exp(y - zeta_nutau) + 1) + 1/(np.exp(y + zeta_nutau) + 1) - 2*1/(np.exp(y) + 1))

            Del_rho_e += weight*Ee*(1/(np.exp(Ee - zeta_e) + 1) + 1/(np.exp(Ee + zeta_e) + 1) - 2*1/(np.exp(Ee) + 1))
            Del_rho_mu += weight*Emu*(1/(np.exp(Emu - zeta_mu) + 1) + 1/(np.exp(Emu + zeta_mu) + 1) - 2*1/(np.exp(Emu) + 1))
            Del_rho_tau += weight*Etau*(1/(np.exp(Etau - zeta_tau) + 1) + 1/(np.exp(Etau + zeta_tau) + 1) - 2*1/(np.exp(Etau) + 1))

            N_nue += weight*(1/(np.exp(y - zeta_nue) + 1) - 1/(np.exp(y + zeta_nue) + 1))
            N_numu += weight*(1/(np.exp(y - zeta_numu) + 1) - 1/(np.exp(y + zeta_numu) + 1))
            N_nutau += weight*(1/(np.exp(y - zeta_nutau) + 1) - 1/(np.exp(y + zeta_nutau) + 1))

            N_e += weight*(1/(np.exp(Ee - zeta_e) + 1) - 1/(np.exp(Ee + zeta_e) + 1))
            N_mu += weight*(1/(np.exp(Emu - zeta_mu) + 1) - 1/(np.exp(Emu + zeta_mu) + 1))
            N_tau += weight*(1/(np.exp(Etau - zeta_tau) + 1) - 1/(np.exp(Etau + zeta_tau) + 1))

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

        #Fixing values of susceptibilities

        TIdx = np.searchsorted(ChiData[:, 0], T)

        Chi2BSlope = (ChiData[TIdx, 1] - ChiData[TIdx - 1, 1]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi2QSlope = (ChiData[TIdx, 2] - ChiData[TIdx - 1, 2]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi11QBSlope = (ChiData[TIdx, 3] - ChiData[TIdx - 1, 3]) / (ChiData[TIdx, 0] - ChiData[TIdx - 1, 0])

        Chi2B = Chi2BSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 1]

        Chi2Q = Chi2QSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 2]

        Chi11QB = Chi11QBSlope * (T - ChiData[TIdx - 1, 0]) + ChiData[TIdx - 1, 3]

        N_B = Chi11QB*zeta_Q + Chi2B*zeta_B
        N_Q = Chi11QB*zeta_B + Chi2Q*zeta_Q



        #Corrections to the total energy density and enetropy

        Delta_rho = 1/(2*np.pi**2)*(Del_rho_nue + Del_rho_numu + Del_rho_nutau + 2*(Del_rho_e + Del_rho_mu + Del_rho_tau)) + 1/2*(Chi2B*zeta_B**2 + 2*Chi11QB*zeta_B*zeta_Q + Chi2Q*zeta_Q**2) + 1/2*T*(Chi2BSlope*zeta_B**2 + 2*Chi11QBSlope*zeta_B*zeta_Q + Chi2QSlope*zeta_Q**2)

        rho = rho_0 + Delta_rho

        s = s_0 + 1/(2*np.pi**2)*(Del_s_nue + Del_s_numu + Del_s_nutau + 2*(Del_s_e + Del_s_mu + Del_s_tau)) + 1/2*T*(Chi2BSlope*zeta_B**2 + 2*Chi11QBSlope*zeta_B*zeta_Q + Chi2QSlope*zeta_Q**2)

        P = s - rho + 1/(2*np.pi**2)*(zeta_nue*N_nue + zeta_numu*N_numu + zeta_nutau*N_nutau + 2*(zeta_e*N_e + zeta_mu*N_mu + zeta_tau*N_tau))  + zeta_B*N_B +zeta_Q*N_Q
        
        #s = s_0
        #rho = rho_0
        #P = s - rho

    else:

        for ni in range(n2):

            y = y_min2 + dy2*ni
            weight = coe_simps2[ni]*y**2
            weight_s = coe_simps2[ni]

            Ee = (y**2 + xe**2)**(1/2)
            Emu = (y**2 + xmu**2)**(1/2)
            Etau = (y**2 + xtau**2)**(1/2)

            Epic = (y**2 + xpic**2)**(1/2)

            Del_rho_nue += weight*y*(1/(np.exp(y - zeta_nue) + 1) + 1/(np.exp(y + zeta_nue) + 1) - 2*1/(np.exp(y) + 1))
            Del_rho_numu += weight*y*(1/(np.exp(y - zeta_numu) + 1) + 1/(np.exp(y + zeta_numu) + 1) - 2*1/(np.exp(y) + 1))
            Del_rho_nutau += weight*y*(1/(np.exp(y - zeta_nutau) + 1) + 1/(np.exp(y + zeta_nutau) + 1) - 2*1/(np.exp(y) + 1))

            Del_rho_e += weight*Ee*(1/(np.exp(Ee - zeta_e) + 1) + 1/(np.exp(Ee + zeta_e) + 1) - 2*1/(np.exp(Ee) + 1))
            Del_rho_mu += weight*Emu*(1/(np.exp(Emu - zeta_mu) + 1) + 1/(np.exp(Emu + zeta_mu) + 1) - 2*1/(np.exp(Emu) + 1))
            Del_rho_tau += weight*Etau*(1/(np.exp(Etau - zeta_tau) + 1) + 1/(np.exp(Etau + zeta_tau) + 1) - 2*1/(np.exp(Etau) + 1))

            Del_rho_pic += weight*Epic*(1/(np.exp(Epic - zeta_pic) - 1) + 1/(np.exp(Epic + zeta_pic) - 1) - 2*1/(np.exp(Epic) - 1))

            N_nue += weight*(1/(np.exp(y - zeta_nue) + 1) - 1/(np.exp(y + zeta_nue) + 1))
            N_numu += weight*(1/(np.exp(y - zeta_numu) + 1) - 1/(np.exp(y + zeta_numu) + 1))
            N_nutau += weight*(1/(np.exp(y - zeta_nutau) + 1) - 1/(np.exp(y + zeta_nutau) + 1))

            N_e += weight*(1/(np.exp(Ee - zeta_e) + 1) - 1/(np.exp(Ee + zeta_e) + 1))
            N_mu += weight*(1/(np.exp(Emu - zeta_mu) + 1) - 1/(np.exp(Emu + zeta_mu) + 1))
            N_tau += weight*(1/(np.exp(Etau - zeta_tau) + 1) - 1/(np.exp(Etau + zeta_tau) + 1))

            N_pic += weight*(1/(np.exp(Epic - zeta_pic) - 1) - 1/(np.exp(Epic + zeta_pic) - 1))

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

            


        s = s_0 + 1/(2*np.pi**2)*(Del_s_nue + Del_s_numu + Del_s_nutau + 2*(Del_s_e + Del_s_mu + Del_s_tau) + Del_s_pic)
        Delta_rho = 1/(2*np.pi**2)*(Del_rho_nue + Del_rho_numu + Del_rho_nutau + 2*(Del_rho_e + Del_rho_mu + Del_rho_tau) + Del_rho_pic)
        rho = rho_0 + Delta_rho
        P = s - rho +  1/(2*np.pi**2)*(zeta_nue*N_nue + zeta_numu*N_numu + zeta_nutau*N_nutau + 2*(zeta_e*N_e + zeta_mu*N_mu + zeta_tau*N_tau + zeta_pic*N_pic))

        #s = s_0
        #rho = rho_0
        #P = s - rho

    return rho, s, P, Delta_rho, rho_nua, rho_a, Del_nua, Del_a, Del_nu_total, Del_c_total, Del_Q_QCD



