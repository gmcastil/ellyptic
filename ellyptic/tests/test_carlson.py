from __future__ import division
import math

import numpy as np
from numpy.testing import assert_equal, assert_approx_equal

import carlson

BOOST_DATA = "./tests/data/boost.npz"
SIG_FIGS = 14
RTOL = 1E-40

class TestBoost():

    def setup(self):
        self.data = np.load(BOOST_DATA)

    def test_RF(self):
        cases = self.data["ellint_rf_data_ipp-ellint_rf_data"]
        for test in cases:
            x, y, z, result = test
            assert_approx_equal(carlson.R_F(x, y, z, rtol=RTOL), result,
                                significant=SIG_FIGS)

    def test_RC(self):
        cases = self.data["ellint_rc_data_ipp-ellint_rc_data"]
        for test in cases:
            x, y, result = test
            assert_approx_equal(carlson.R_C(x, y, rtol=RTOL), result,
                                significant=SIG_FIGS)

    def test_RJ(self):
        cases = self.data["ellint_rj_data_ipp-ellint_rj_data"]
        for test in cases:
            x, y, z, p, result = test
            assert_complex_equal(carlson.R_J(x, y, z, p, rtol=RTOL), result,
                                 significant=SIG_FIGS)

    # def test_RD(self):
    #     cases = self.data["ellint_rd_data_ipp-ellint_rd_data"]
    #     for test in cases:
    #         x, y, z, result = test
    #         assert_approx_equal(carlson.R_D(x, y, z, rtol=RTOL), result,
    #                             significant=SIG_FIGS)

    def teardown(self):
        pass

class TestDiscretes():

    def test_RF(self):
        result = carlson.R_F(1, 2, 0, rtol=RTOL)
        assert_approx_equal(result,
                            1.3110287771461,
                            significant=SIG_FIGS)

        result = carlson.R_F(1j, -1j, 0, rtol=RTOL)
        assert_complex_equal(result,
                             1.8540746773014,
                             significant=SIG_FIGS)

        result = carlson.R_F(1j-1, 1j, 0, rtol=RTOL)
        assert_complex_equal(result,
                             0.79612586584234-1.2138566698365j,
                             significant=SIG_FIGS)

        result = carlson.R_F(2, 3, 4, rtol=RTOL)
        assert_complex_equal(result,
                             0.58408284167715,
                             significant=SIG_FIGS)

        result = carlson.R_F(1j, -1j, 2, rtol=RTOL)
        assert_complex_equal(result,
                             1.0441445654064,
                             significant=SIG_FIGS)

        result = carlson.R_F(1j-1, 1j, 1-1j, rtol=RTOL)
        assert_complex_equal(result,
                             0.93912050218619-0.53296252018635j,
                             significant=SIG_FIGS)

    def test_RC(self):
        # Some of these have closed form solutions including pi or various
        # natural logs
        result = carlson.R_C(0, 1/4, rtol=RTOL)
        assert_complex_equal(result,
                             math.pi,
                             significant=SIG_FIGS)

        result = carlson.R_C(9/4, 2, rtol=RTOL)
        assert_complex_equal(result,
                             math.log(2),
                             significant=SIG_FIGS)
        """
        A lot of this can be refactored using something like functools.partial

        """
        result = carlson.R_C(0, 1j, rtol=RTOL)
        assert_complex_equal(result,
                             (1-1j)*1.1107207345396,
                             significant=SIG_FIGS)

        result = carlson.R_C(-1j, 1j, rtol=RTOL)
        assert_complex_equal(result,
                             1.2260849569072-0.34471136988768j,
                             significant=SIG_FIGS)

        result = carlson.R_C(1/4, -2, rtol=RTOL)
        assert_complex_equal(result,
                             (1/3)*math.log(2),
                             significant=SIG_FIGS)

        result = carlson.R_C(1j, -1, rtol=RTOL)
        assert_complex_equal(result,
                             0.77778596920447+0.19832484993429j,
                             significant=SIG_FIGS)

    def test_RJ(self):
        result = carlson.R_J(0, 1, 2, 3, rtol=RTOL)
        assert_complex_equal(result,
                             0.77688623778582,
                             significant=SIG_FIGS)

def assert_complex_equal(a, b, **kwargs):
    assert_approx_equal(a.real, b.real, **kwargs)
    assert_approx_equal(a.imag, b.imag, **kwargs)
