"""
The effective number of relativistic degree of freedom for energy density and entropy density
The fitting formula are adopted from K. Saikawa and S. Shirai, arXiv: 1803.01038.
"""


import numpy as np
from numba import jit

# g below 120 MeV

#constants

me = 0.511 #MeV

mmu = 105.6 #MeV

mpi0 = 135 #MeV

mpic = 140 #MeV

m1 = 500 #MeV

m2 = 770 #MeV

m3 = 1200 #MeV

m4 = 2000 #MeV


@jit(nopython=True,nogil=True,fastmath=True)
def grho_below120MeV(T):

    u_e = me/T

    u_mu = mmu/T

    u_pi0 = mpi0/T

    u_pic = mpic/T

    u1 = m1/T

    u2 = m2/T

    u3 = m3/T

    u4 = m4/T


    g = 2.030 + 1.353*Sfit(u_e)**(4/3) + 3.495*frho(u_e) + 3.446*frho(u_mu) \
          + 1.05*brho(u_pi0) + 2.08*brho(u_pic) + 4.165*brho(u1) + 30.55*brho(u2) + 89.4*brho(u3) + 8209*brho(u4)


    return g


@jit(nopython=True,nogil=True,fastmath=True)
def gs_below120MeV(T):

    u_e = me/T

    u_mu = mmu/T

    u_pi0 = mpi0/T

    u_pic = mpic/T

    u1 = m1/T

    u2 = m2/T

    u3 = m3/T

    u4 = m4/T

    g = 2.008 + 1.923*Sfit(u_e) + 3.442*fs(u_e) + 3.468*fs(u_mu) \
          + 1.034*bs(u_pi0) + 2.068*bs(u_pic) + 4.16*bs(u1) + 30.55*bs(u2) + 90*bs(u3) + 6209*bs(u4)

    return g

@jit(nopython=True,nogil=True,fastmath=True)
def dgrhodT_below120MeV(T):

    u_e = me/T

    u_mu = mmu/T

    u_pi0 = mpi0/T

    u_pic = mpic/T

    u1 = m1/T

    u2 = m2/T

    u3 = m3/T

    u4 = m4/T

    fac = - 1/(T**2)

    dgdz = fac*(me*1.353*4/3*Sfit(u_e)**(1/3)*dSfitdx(u_e) + me*3.495*dfrhodx(u_e) + mmu*3.446*dfrhodx(u_mu) \
          + mpi0*1.05*dbrhodx(u_pi0) + mpic*2.08*dbrhodx(u_pic) + m1*4.165*dbrhodx(u1) + m2*30.55*dbrhodx(u2) + m3*89.4*dbrhodx(u3) + m4*8209*dbrhodx(u4))

    return dgdz


@jit(nopython=True,nogil=True,fastmath=True)
def Sfit(u):
    
    Sfit = 1 + 7/4*np.exp(-1.0419*u)*(1 + 1.034*u + 0.456426*u**2 + 0.0595249*u**3)

    return Sfit

@jit(nopython=True,nogil=True,fastmath=True)
def dSfitdx(u):
    
    dSfitdx = -1.0419*7/4*np.exp(-1.0419*u)*(1 + 1.034*u + 0.456426*u**2 + 0.0595249*u**3) + 7/4*np.exp(-1.0419*u)*(1.034 + 2*0.456426*u + 3*0.0595249*u**2)

    return dSfitdx

@jit(nopython=True,nogil=True,fastmath=True)
def frho(u):

    frho = np.exp(-1.04855*u)*(1 + 1.03757*u + 0.508630*u**2 + 0.0893988*u**3)

    return frho

@jit(nopython=True,nogil=True,fastmath=True)
def dfrhodx(u):

    dfrhodx = -1.04855*np.exp(-1.04855*u)*(1 + 1.03757*u + 0.508630*u**2 + 0.0893988*u**3) + np.exp(-1.04855*u)*(1.03757 + 2*0.508630*u + 3*0.0893988*u**2)

    return dfrhodx

@jit(nopython=True,nogil=True,fastmath=True)
def brho(u):

    brho = np.exp(-1.03149*u)*(1 + 1.03317*u + 0.398264*u**2 + 0.0648056*u**3)

    return brho

@jit(nopython=True,nogil=True,fastmath=True)
def dbrhodx(u):

    dbrhodx = -1.03149*np.exp(-1.03149*u)*(1 + 1.03317*u + 0.398264*u**2 + 0.0648056*u**3) + np.exp(-1.03149*u)*(1.03317 + 2*0.398264*u + 3*0.0648056*u**2)

    return dbrhodx

@jit(nopython=True,nogil=True,fastmath=True)
def fs(u):

    fs = np.exp(-1.04190*u)*(1 + 1.03400*u + 0.456426*u**2 + 0.0506182*u**3)

    return fs

@jit(nopython=True,nogil=True,fastmath=True)
def bs(u):

    bs = np.exp(-1.03365*u)*(1 + 1.03397*u + 0.342548*u**2 + 0.0506182*u**3)

    return bs



#g above 120 MeV

#constant

a0 =1

a1 = 1.11724

a2 = 3.12672e-1

a3 = -4.68049e-2

a4 = -2.65004e-2

a5 = -1.19760e-3

a6 = 1.82812e-4

a7 = 1.36436e-4

a8 = 8.55051e-5

a9 = 1.22840e-5

a10 = 3.82259e-7

a11 = -6.87035e-9



b0 = 1.43382e-2

b1 = 1.37559e-2

b2 = 2.92108e-3

b3 = -5.38533e-4

b4 = -1.62496e-4

b5 = -2.87906e-5

b6 = -3.84278e-6

b7 = 2.78776e-6

b8 = 7.40342e-7

b9 = 1.17210e-7

b10 = 3.72499e-9

b11 = -6.74107e-11



c0 = 1

c1 = 6.07869e-1

c2 = -1.54485e-1

c3 = -2.24034e-1

c4 = -2.82147e-2

c5 = 2.90620e-2

c6 = 6.86778e-3

c7 = -1.00005e-3

c8 = -1.69104e-4

c9 = 1.06301e-5

c10 = 1.69528e-6

c11 = -9.33311e-8




d0 = 7.07388e1

d1 = 9.18011e1

d2 = 3.31892e1

d3 = -1.39779e0

d4 = -1.52558e0

d5 = -1.97857e-2

d6 = -1.60146e-1

d7 = 8.22615e-5

d8 = 2.02651e-2

d9 = -1.82134e-5

d10 = 7.83943e-5

d11 = 7.13518e-5



@jit(nopython=True,nogil=True,fastmath=True)
def grho_above120MeV(T):

    t = np.log(T*1e-3)

    a = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]
    b = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11]

    sum_a = 0
    sum_b = 0

    for i in range(12):

        sum_a += a[i]*t**i
        sum_b += b[i]*t**i


    g = sum_a/sum_b

    return g


@jit(nopython=True,nogil=True,fastmath=True)
def gs_above120MeV(T):

    t = np.log(T*1e-3)

    c = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
    d = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11]

    sum_c = 0
    sum_d = 0

    for i in range(12):

        sum_c += c[i]*t**i
        sum_d += d[i]*t**i


    g = grho_above120MeV(T)/(1 + sum_c/sum_d)


    return g

@jit(nopython=True,nogil=True,fastmath=True)
def dgrhodT_above120MeV(T):

    t = np.log(T*1e-3)

    a = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]
    b = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11]

    sum_a = 0
    sum_b = 0
    sum_a2 = 0
    sum_b2 = 0

    for i in range(12):

        sum_a += a[i]*t**i
        sum_b += b[i]*t**i
        
    for i in range(11):

        sum_a2 += (i+1)*a[i+1]*t**i
        sum_b2 += (i+1)*b[i+1]*t**i


    dgdx = (sum_a2*sum_b - sum_a*sum_b2)/sum_b**2*1/T

    return dgdx
