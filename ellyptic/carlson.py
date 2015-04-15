# pylint: disable=C0103, R0914
"""
Numerical computation of Carlson symmetric forms of elliptic integrals

Implemented using algorithms described here:

http://arxiv.org/abs/math/9409227

"""

from numpy.lib import scimath

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
    Q = ((3*rtol) ** (-1.0/6.0)) * M

    A = _A_gen(x, y, z)
    for n, Am in enumerate(A):
        if 4**(-n) * Q < abs(Am):
            X = (A0 - x) / (Am * 4**n)
            Y = (A0 - y) / (Am * 4**n)
            Z = -X-Y
            E2 = X * Y - Z**2
            E3 = X * Y * Z
            result = (1/scimath.sqrt(Am)) * (1 - E2*(1.0/10) + E3*(1.0/14)
                                             + (E2**2)*(1.0/24) - (E2*E3)*(3.0/44))
            return result

def R_C(x, y, rtol=2e-4):
    r"""Computes a degenerate case of `R_J`

    .. math:: R_C(x, y) = \frac{1}{2}\int_0^\infty
                                      (t + x)^{1-\frac{1}{2}}(t + y)^{-1} dt

    """
    factor = 1
    if _is_neg_real(y):
        factor = scimath.sqrt(x / (x + abs(y)))
        x = x + abs(y)
        y = abs(y)

    A0 = (x + 2*y) / 3.0
    Q = (3*rtol)**(-1.0/8.0)*abs(A0 - x)
    A = _A_gen(x, y, y)
    for n, Am in enumerate(A):
        if 4**(-n) * Q < abs(Am):
            s = (y - A0) / (4**n * Am)
            result = (1/scimath.sqrt(Am)) * (1 + s**2 * (3.0/10) + s**3 * (1.0/7)
                                             + s**4 * (3.0/8) + s**5 * (9.0/22)
                                             + s**6 * (159.0/208) + s**7 * (9.0/8))
            return factor * result

def R_J(x, y, z):
    r"""Computes the symmetric integral of the third kind

    .. math:: R_J(x, y, z) =
              \frac{1}{3}\int_0^\infty [(t + x)(t + y)(t + z)]^{-\frac{1}{2}}
                                       (t + p)^{-1}dt

    """
    pass

def R_D(x, y, z):
    r"""Computes a degenerate case of `R_F`

    .. math::

    """



def _A_gen(x, y, z):
    """Generates `A_m` values

    """
    xm, ym, zm = x, y, z
    Am = (xm + ym + zm) / 3
    yield Am
    while True:
        lm = _lambda_m(xm, ym, zm)
        Am = (Am + lm) / 4
        xm = (xm + lm) / 4
        ym = (ym + lm) / 4
        zm = (zm + lm) / 4
        yield Am

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
