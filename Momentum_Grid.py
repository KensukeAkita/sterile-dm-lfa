import numpy as np
from Constants import * 
from QCD_GlobalParameters import * 

#Setting for linear momentum grids

n = 10001 #number of grids for sterile neutrinos 
#The actual number of momentum grids is n-1. 
#For Le=-Lmu=0.1, Ltau=0, we use n=2e5.

y_max = 16
y_min = 0.1

y_sterile = np.zeros(n)
dy_sterile = np.zeros(n)

for ni in range(n):

    if (ni == 0) or (ni == n-1):

        fac = 1/3
        
    elif (ni % 2 == 1):
        
        fac = 4/3

    else:

        fac = 2/3

    dy_sterile[ni] = fac*(y_max - y_min)/(n-1)

    dy = (y_max - y_min)/(n-1)

    y_sterile[ni] = y_min + dy*ni
    
    
    
#Grids for the table of neutrino reaction rates

ni_Table = 100

y_Table = np.linspace(y_min,y_max,ni_Table)

#Array between TQCD and Tcut
    
nT_intermedi = 50
T_intermedi = np.zeros(nT_intermedi)

for nT in range(nT_intermedi):
    
    T_intermedi[nT] = LambdaQCD + (Tcut-LambdaQCD)/(nT_intermedi - 1)*nT
    
#Array below TQCD
    
nT_low = 100 
T_low = np.logspace(1,np.log10(LambdaQCD),nT_low)

#Array above Tcut
    
nT_high = 200
T_high = np.logspace(np.log10(Tcut),4,nT_high)

#Total array
    
nT_Table = nT_low + nT_intermedi + nT_high

T_Table = np.concatenate((T_low,T_intermedi,T_high))



n1 = 41 #bins for the integrals in the neutrino interaction rate, n1 must be an odd number because we use the Simpson method

y_max1 = 20
y_min1 = 0.01

dy1 = (y_max1 - y_min1)/(n1-1)

n2 = 41 #bins for lepton asymmetry redistribution

y_max2 = 20
y_min2 = 0.01
dy2 = (y_max2 - y_min2)/(n2-1)

n3 = 21 #bins for QCD corrections

y_max3 = 80
y_min3 = 0.01
dy3 = (y_max3 - y_min3)/(n3-1)



#Coefficients for the Simpson method in integrals
        
coe_simps = np.zeros(n1)
        
for ni in range(n1):

    if (ni == n1-1) or (ni == 0):
                    
        coe_simps[ni] = 1/3*dy1
                    
    elif (ni % 2 == 1):
                    
        coe_simps[ni] = 4/3*dy1
                    
    else:
            
        coe_simps[ni] = 2/3*dy1

        
coe_simps2 = np.zeros(n2)
        
for ni in range(n2):

    if (ni == n2-1) or (ni == 0):
                    
        coe_simps2[ni] = 1/3*dy2
                    
    elif (ni % 2 == 1):
                    
        coe_simps2[ni] = 4/3*dy2
                    
    else:
            
        coe_simps2[ni] = 2/3*dy2

coe_simps3 = np.zeros(n1)
        
for ni in range(n3):

    if (ni == n3-1) or (ni == 0):
                    
        coe_simps3[ni] = 1/3*dy3
                    
    elif (ni % 2 == 1):
                    
        coe_simps3[ni] = 4/3*dy3
                    
    else:
            
        coe_simps3[ni] = 2/3*dy3