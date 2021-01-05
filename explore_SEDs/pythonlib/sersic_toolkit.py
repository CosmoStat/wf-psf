import numpy as np
from math import gamma

def bn(n):
    """ Compute b_n coefficient, the normalization constant inside the Sersic light intensity
    profile. In truth, b_n is the number such that Gamma(2n) = 2gamma(2n,b_n), where Gamma
    is the (complete) Gamma function and gamma the lower incomplete gamma function (see eg
    Graham & Driver, 2005). Here we use several approximate analytical expressions depending
    on the n value."""
    if n<=0.36: # MacArthur, Courteau and Holtzmann, 2005
        monomials = [.01945, -.8902, 10.95, -19.67, 13.43]
        return np.sum([mono * n**j for j,mono in enumerate(monomials)])
    else:       # Ciotti & Bertin, 1999
        monomials = [-1./3, 4./405, 46./25515, 131./1148175, 2194697./30690717750]
        poly = [2.*n] + [mono * 1./n**j for j,mono in enumerate(monomials)]
        return np.sum(poly)

def hlr_to_R2(hlr, n):
    """ Convert Sercic half light radius R_e to 'size' R^2 in the sense of quadrupole moments."""
    return hlr**2 / bn(n)**(2.*n) * gamma(4.*n) / gamma(2.*n)
