import numpy as np

#Constants

alpha   =   7.2973525*10**(-3) #Fine structure constant
e       =   (4*np.pi*alpha)**(1/2) # electromagnetic couping

sW      =   (0.231)**(1/2) #Weak mixing angle
gL      =   0.731 #1/2+ sW**2 #0.731
gLtilde =   -0.269 #-1/2 + sW**2 #-0.269
gR      =   0.231 #sW**2 #0.231
GF      =   1.1663787*10**(-11) #MeV**(-2) #Fermi coupling constant
mW      =   80.379*10**3        #MeV #W boson mass
mZ      =   mW*(1-sW**2)**(1/2)    #MeV #W boson mass
G       =   6.70833*10**(-45) #MeV**(-2) #Gravitational constant 
mpl     =   (1/G)**(1/2) #Planck mass

#Charged lepton masses

me      =   0.5109989 #MeV #electron mass
mmu     =   105.66 #MeV #muon mass
mtau    =   1776.86 #MeV #tau mass

#quark masses (check!)

mu = 2.3 
md = 4.8
mc = 1275
ms = 95

Nf = 3
Nc = 3
dA = Nc**2-1
TF = Nf/2
CA = Nc
CF = (Nc**2-1)/(2*Nc)
LambdaMS = 200 #MeV
muref = 2000 #MeV
GammaE = 0.57721 #Euler-Mascheroni Constant

#CKM matrix

Vud = 0.97373
Vus = 0.2243
Vcd = 0.221
Vcs = 0.975

#hadron masses (check!)

mpi0 = 134.98
mpic = 139.57
mK0 = 497.614
mKc = 493.68
meta = 547.85
mrho0 = 775.26
mrhoc = 775.26
mKc892 = 891.66
momega = 782.65

mp = 938.27

mn = 939.56



