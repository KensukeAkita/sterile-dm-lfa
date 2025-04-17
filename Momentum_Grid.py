import numpy as np
from Constants import * 
from Sterile_Nu_Parameters import * 

#Setting for linear momentum grid

n = 1000 #number of bins for sterile neutrinos
#The actual momentum grid is n-1. 

y_max = 16
y_min = 0.1
delta = 1/n*np.log10(y_max/y_min)

#vectorized momentum grid

y_max2 = np.log10(y_max)
y_min2 = np.log10(y_min)

#y2 = np.logspace(y_min2,y_max2,n)

y2 = np.zeros(n)
dy2 = np.zeros(n)

for ni in range(n):

    if (ni == 0) or (ni == n-1):

        fac = 1/2

    else:

        fac = 1

    #y2[ni] = y_min*10**(delta*ni)
    #dy2[ni] = fac*(y_min*10**(delta*ni) - y_min*10**(delta*(ni-1)))

    dy2[ni] = fac*(y_max - y_min)/(n-1)

    dy = (y_max - y_min)/(n-1)

    y2[ni] = y_min + dy*ni
    
#Reation rate table grid

#nT_Table = 300
ni_Table = 100

#T_Table = np.logspace(1,4,nT_Table)

#y_Table = np.logspace(y_min2,y_max2,ni_Table)

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





n1 = 41 #bins for the collision term, n1 must be an odd number because we use the Simpson method
#n2 = 41

y_max1 = 20
y_min1 = 0.01
#dy = (y_max - y_min)/(n-1)

dy1 = (y_max1 - y_min1)/(n1-1)
#dy2 = (y_max - y_min)/(n2-1)

n3 = 41 #bins for lepton asymmetry redistribution

y_max3 = 20
y_min3 = 0.01
dy3 = (y_max3 - y_min3)/(n3-1)

n4 = 21 #bins for QCD corrections

y_max4 = 80
y_min4 = 0.01
dy4 = (y_max4 - y_min4)/(n4-1)


#Coefficients for Simpson method in integrals
        
coe_simps = np.zeros(n1)
        
for ni in range(n1):

    if (ni == n1-1) or (ni == 0):
                    
        coe_simps[ni] = 1/3*dy1
                    
    elif (ni % 2 == 1):
                    
        coe_simps[ni] = 4/3*dy1
                    
    else:
            
        coe_simps[ni] = 2/3*dy1

        
coe_simps3 = np.zeros(n3)
        
for ni in range(n3):

    if (ni == n3-1) or (ni == 0):
                    
        coe_simps3[ni] = 1/3*dy3
                    
    elif (ni % 2 == 1):
                    
        coe_simps3[ni] = 4/3*dy3
                    
    else:
            
        coe_simps3[ni] = 2/3*dy3

coe_simps4 = np.zeros(n1)
        
for ni in range(n4):

    if (ni == n4-1) or (ni == 0):
                    
        coe_simps4[ni] = 1/3*dy4
                    
    elif (ni % 2 == 1):
                    
        coe_simps4[ni] = 4/3*dy4
                    
    else:
            
        coe_simps4[ni] = 2/3*dy4