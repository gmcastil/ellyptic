# pylint: disable=C0103, R0914
"""
Numerical computation of Carlson symmetric forms of elliptic integrals

Implemented using algorithms described here:

http://arxiv.org/abs/math/9409227

"""

from __future__ import division
from numpy.lib import scimath
import pdb

def R_F(x, y, z, rtol=3e-4):
    r"""Computes the symmetric integral of the first kind

    .. math:: R_F(x, y, z) =
              \frac{1}{2}\int_0^\infty [(t + x)(t + y)(t + z)]^\frac{1}{2}dt

    """
    # Properly deal with the degenerate case
    if y == z:
        return R_C(x, y)

    A0 = (x + y + z) / 3.0
    M = max(abs(A0 - x), abs(A0 - y), abs(A0 - z))
    Q = ((3*rtol)**(-1.0/6.0)) * M

    A = _A1_gen(x, y, z)
    for m, Am in enumerate(A):
        if 4**(-m) * Q < abs(Am):
            X = (A0 - x) / (Am * 4**m)
            Y = (A0 - y) / (Am * 4**m)
            Z = -X-Y
            E2 = X * Y - Z**2
            E3 = X * Y * Z
            result = (1/scimath.sqrt(Am)) * (1 - E2*(1/10) + E3*(1/14)
                                             + (E2**2)*(1/24) - (E2*E3)*(3/44))
            return result

def R_C(x, y, rtol=3e-4):
    r"""Computes a degenerate case of `R_J`

    .. math:: R_C(x, y) = \frac{1}{2}\int_0^\infty
                                      (t + x)^{1-\frac{1}{2}}(t + y)^{-1} dt

    """
    factor = 1
    if _is_neg_real(y):
        factor = scimath.sqrt(x / (x + abs(y)))
        x = x + abs(y)
        y = abs(y)

    A0 = (x + 2*y) / 3
    Q = (3*rtol)**(-1/8)*abs(A0 - x)
    A = _A1_gen(x, y, y)
    for m, Am in enumerate(A):
        if 4**(-m) * Q < abs(Am):
            s = (y - A0) / (4**m * Am)
            result = (1/scimath.sqrt(Am)) * (1 + s**2 * (3/10) + s**3 * (1/7)
                                             + s**4 * (3/8) + s**5 * (9/22)
                                             + s**6 * (159/208) + s**7 * (9/8))
            return factor * result

def R_J(x, y, z, p, rtol=3e-4):
    r"""Computes the symmetric integral of the third kind

    .. math:: R_J(x, y, z) =
              \frac{1}{3}\int_0^\infty [(t + x)(t + y)(t + z)]^{-\frac{1}{2}}
                                       (t + p)^{-1}dt

    """
    A0 = (x + y + z + 2*p) / 5
    delta = (p - x)*(p - y)*(p - z)
    M = max(abs(A0 - x), abs(A0 - y), abs(A0 - z), abs(A0 - p))
    Q = ((rtol/4)**(-1/6)) * M

    A = _A2_gen(x, y, z, p)
    for m, Am in enumerate(A):
        if 4**(-m) * Q < abs(Am):
            X = (A0 - x) / (4**m * Am)
            Y = (A0 - y) / (4**m * Am)
            Z = (A0 - z) / (4**m * Am)
            P = (-X-Y-Z) / 2
            n = m
            An = Am
            break

    E2 = X*Y + X*Z + Y*Z - 3*P**2
    E3 = X*Y*Z + 2*E2*P + 4*P**3
    E4 = (2*X*Y*Z + E2*P + 3*P**3) * P
    E5 = X*Y*Z*P**2

    result = 4**(-n)*An**(-3/2) * (1 - (3/14)*E2 + (1/6)*E3 + (9/88)*E2**2
                                   -(3/22)*E4 - (9/52)*E2*E3 + (3/26)*E5)
    # Compute the contribution from the R_C terms
    contrib = 0
    d = _d_gen(x, y, z, p)
    for m, dm in enumerate(d):
        em = (4**(-3*m) * delta) / dm**2
        contrib += (4**(-m)/dm)*R_C(1, 1+em, rtol)
        if n == 0 or m == n - 1:
            break

    return result + 6*contrib

def R_D(x, y, z, rtol=3e-4):
    r"""Computes a degenerate case of `R_F`

    .. math::

    """
    pass

def _d_gen(x, y, z, p):
    """Generates `d_m` values"""
    xm, ym, zm, pm = x, y, z, p
    dm = ((scimath.sqrt(pm) + scimath.sqrt(xm))
          * (scimath.sqrt(pm) + scimath.sqrt(ym))
          * (scimath.sqrt(pm) + scimath.sqrt(zm)))
    yield dm
    while True:
        lm = _lambda_m(xm, ym, zm)
        xm = (xm + lm) / 4
        ym = (ym + lm) / 4
        zm = (zm + lm) / 4
        pm = (pm + lm) / 4
        dm = ((scimath.sqrt(pm) + scimath.sqrt(xm))
              * (scimath.sqrt(pm) + scimath.sqrt(ym))
              * (scimath.sqrt(pm) + scimath.sqrt(zm)))
        yield dm

def _A1_gen(x, y, z):
    """Generates `A_m` values for R_F and R_C"""
    xm, ym, zm = x, y, z
    Am = (xm + ym + zm) / 3
    yield Am
    while True:
        lm = _lambda_m(xm, ym, zm)
        Am = (Am + lm) / 4
        yield Am
        xm = (xm + lm) / 4
        ym = (ym + lm) / 4
        zm = (zm + lm) / 4

def _A2_gen(x, y, z, p):
    """Generates `A_m` for R_J and R_D"""
    xm, ym, zm, pm = x, y, z, p
    Am = (xm + ym + zm + 2*pm) / 5
    yield Am
    while True:
        lm = _lambda_m(xm, ym, zm)
        Am = (Am + lm) / 4
        yield Am
        xm = (xm + lm) / 4
        ym = (ym + lm) / 4
        zm = (ym + lm) / 4
        pm = (pm + lm) / 4


def _lambda_m(x, y, z):
    r"""

    .. math:: \lambda_m = \sqrt{x_m}\sqrt{y_m} + \sqrt{x_m}\sqrt{y_m}
                                    + \sqrt{y_m}\sqrt{z_m}

    Note that :math:\sqrt{x_m}\sqrt{y_m} is chosen instead of
    :math:\sqrt{x_m y_m} to avoid problems due to the branch cut

    """
    a, b, c = scimath.sqrt([x, y, z])
    lm = a*b + a*c + b*c
    return lm

def _is_neg_real(x):
    if x.imag == 0 and x.real < 0:
        return True
    else:
        return False
