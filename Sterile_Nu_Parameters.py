

#Sterile Nu parameters

Uas = (1.70*10**(-14))**(1/2)
ms = 50.11*1e-3 #keV->MeV 

Le_ini = 1e-3
Lmu_ini = 1e-3
Ltau_ini = 1e-3
flavor = 'e' #'e' or 'mu' or 'tau'

#Lepton asymmetry mixing with sterile neutrinos

if (flavor == 'e'):

    La_ini = Le_ini
    
elif(flavor == 'mu'):
    
    La_ini = Lmu_ini
    
else:
    
    La_ini = Ltau_ini


