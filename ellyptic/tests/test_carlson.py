import numpy as np
from numpy.testing import assert_equal, assert_approx_equal

import carlson

BOOST_DATA = "./tests/data/boost.npz"

class TestBoost():

    def setup(self):
        self.data = np.load(BOOST_DATA)

    def test_RF(self):
        cases = self.data["ellint_rf_data_ipp-ellint_rf_data"]
        for test in cases:
            x, y, z, result = test
            assert_approx_equal(carlson.R_F(x, y, z, rtol=1E-40), result,
                                significant=14)

    def test_RC(self):
        cases = self.data["ellint_rc_data_ipp-ellint_rc_data"]
        for test in cases:
            x, y, result = test
            assert_approx_equal(carlson.R_C(x, y, rtol=1E-40), result,
                                significant=14)

    # def test_RJ(self):
    #     cases = self.data["ellint_rj_data_ipp-ellint_rj_data"]
    #     for test in cases:
    #         x, y, z, p, result = test
    #         assert_approx_equal(carlson.R_J(x, y, z, p, rtol=1E-40), result,
    #                             significant=14)

    # def test_RD(self):
    #     cases = self.data["ellint_rd_data_ipp-ellint_rd_data"]
    #     for test in cases:
    #         x, y, z, result = test
    #         assert_approx_equal(carlson.R_D(x, y, z, rtol=1E-40), result,
    #                             significant=14)

    def teardown(self):
        pass

class TestDiscretes():

    def test_RF(self):
        assert_approx_equal(carlson.R_F(1, 2, 0, rtol=1E-40),
                            1.3110287771461,
                            significant=14)
        result = carlson.R_F(1j, -1j, 0, rtol=1E-40)
        assert_approx_equal(result.real, 1.8540746773014, significant=14)
        assert_approx_equal(result.imag, 0.0, significant=14)
        # assert_approx_equal(carlson.R_F(1j, -1j, 0, rtol=1E-40),
        #                     1.8540746773014,
        #                     significant=14)
        # result = carlson.R_F(1j, -1j, 0, rtol=1E-40)
        # assert_approx_equal(result.real, 1.8540746773014, significant=14)
        # assert_approx_equal(result.imag, 0.0, significant=14)

        # result = carlson.R_F(1j-1, 1j, 0, rtol=1E-40)
        # assert_approx_equal(result.real, 0.79612586584234, significant=14)
        # assert_approx_equal(result.imag, -1.2138566698365j, significant=14)

        # result = carlson.R_F(2, 3, 4, rtol=1E-40)
        # assert_complex_equal(result, 0.58408284167715)
        # result = carlson.R_F(1j, -1j, 2, rtol=1E-40)
        # assert_complex_equal(result, 1.0441445654064)
        # result = carlson.R_F(1j-1, 1j, 1-1j, rtol=1E-40)
        # assert_complex_equal(result, 0.93912050218619-0.53296252018635j)

    # def test_RC(self):
    #     pass

    # def test_RJ(self):
    #     pass

    # def test_RD(self):
    #     pass


